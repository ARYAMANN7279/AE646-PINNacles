"""
Stage 3 training framework (FNO / MLP on the real PDEBench Darcy data).

One run = one JSON spec -> results/stage3/runs/<name>/metrics.json (+ curve.json, optional best_model.pt).
Everything the Stage 3 study varies is an argument, so a single script covers the whole study:

  model     : --model fno|mlp   --width --modes --layers   (FNO)   /   --hidden (MLP)
  data      : --res {32,64,128} training resolution (stride-subsampled from the 128x128 arrays)
              --n-train N (first 900 = the Stage 2 training set; N > 900 draws further real samples
              from the never-used 8,800 "extra" pool; validation (100) and test (200) never change)
              --aug none|d4   exact D4 symmetry augmentation (flips/rotations), applied on the 128x128
              arrays BEFORE subsampling so augmented samples have the same grid layout as real ones
  training  : --steps total optimiser steps (fixed budget)  --batch --lr --wd --loss mse|rel_l2
              --sched legacy_step|cosine   (legacy_step reproduces the Stage 2 per-step cosine cycling
              exactly; cosine = one warm-up + single cosine decay over all steps)
  eval      : val/test relative L2 in physical units at the training resolution, at any --eval-res
              (zero-shot resolution transfer for the FNO), and optional test-time D4 symmetrisation --tta

Selection protocol: the checkpoint with the lowest VALIDATION error is evaluated once on the TEST set.

Run:  python src/stage3_train.py --name demo --model fno --steps 2000
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import get_model, count_parameters

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT_ROOT = ROOT / "results" / "stage3" / "runs"


# ── D4 symmetry group acting on the last two dims ─────────────────────────────
def d4(x, t):
    """Apply element t (0..7) of the dihedral group: rotation by 90*(t%4) degrees, then a flip if t >= 4."""
    x = torch.rot90(x, t % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if t >= 4 else x


def d4_inv(x, t):
    x = torch.flip(x, dims=(-1,)) if t >= 4 else x
    return torch.rot90(x, -(t % 4), dims=(-2, -1))


# ── data ──────────────────────────────────────────────────────────────────────
class Data:
    def __init__(self, n_train, device):
        K = np.load(CACHE / "kappa128.npy", mmap_mode="r")
        U = np.load(CACHE / "u128.npy", mmap_mode="r")
        sp = np.load(CACHE / "splits.npz")
        self.x = torch.tensor(sp["x"], dtype=torch.float32, device=device)     # cell-centre coordinates (128,)
        train_idx = sp["train"] if n_train <= len(sp["train"]) else np.concatenate([sp["train"], sp["extra"][: n_train - len(sp["train"])]])
        train_idx = train_idx[:n_train]
        get = lambda idx: (torch.tensor(np.stack([K[i] for i in idx]), device=device),
                           torch.tensor(np.stack([U[i] for i in idx]), device=device))
        self.k_tr, self.u_tr = get(train_idx)
        self.k_va, self.u_va = get(sp["val"])
        self.k_te, self.u_te = get(sp["test"])
        self.device = device

    def coords(self, res):
        s = 128 // res
        xs = self.x[::s]
        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        return X, Y

    def make(self, k128, u128, res, stats):
        """Stride-subsample to `res`, standardise, add coordinate channels -> (B,res,res,3), (B,res,res,1)."""
        s = 128 // res
        k = (k128[:, ::s, ::s] - stats["k_mean"]) / stats["k_std"]
        u = (u128[:, ::s, ::s] - stats["u_mean"]) / stats["u_std"]
        X, Y = self.coords(res)
        B = k.shape[0]
        inp = torch.stack([k, X.expand(B, -1, -1), Y.expand(B, -1, -1)], dim=-1)
        return inp, u.unsqueeze(-1)


def make_stats(data, res):
    s = 128 // res
    k, u = data.k_tr[:, ::s, ::s], data.u_tr[:, ::s, ::s]
    return {"k_mean": float(k.mean()), "k_std": float(k.std()), "u_mean": float(u.mean()), "u_std": float(u.std())}


def rel_l2_phys(pred, target, stats):
    """Per-sample relative L2 error in PHYSICAL units (pred/target are standardised)."""
    p = pred * stats["u_std"] + stats["u_mean"]
    t = target * stats["u_std"] + stats["u_mean"]
    d = (p - t).reshape(p.shape[0], -1)
    return d.norm(dim=1) / t.reshape(t.shape[0], -1).norm(dim=1)


@torch.no_grad()
def predict(model, data, k128, res, stats, bs=100):
    outs = []
    for i in range(0, k128.shape[0], bs):
        kb = k128[i:i + bs]
        inp, _ = data.make(kb, torch.zeros_like(kb), res, stats)
        outs.append(model(inp))
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, data, k128, u128, res, stats, tta=False):
    """Per-sample physical rel-L2 at resolution `res`. tta: average over the symmetry group that maps the
    stride grid onto itself (all 8 elements at res=128; identity + transpose otherwise)."""
    model.eval()
    _, tgt = data.make(k128, u128, res, stats)
    if not tta:
        return rel_l2_phys(predict(model, data, k128, res, stats), tgt, stats).cpu().numpy()
    group = list(range(8)) if res == 128 else ["id", "T"]
    acc = 0
    for g in group:
        if g == "id":
            kk = k128; inv = lambda z: z
        elif g == "T":
            kk = k128.transpose(-1, -2).contiguous(); inv = lambda z: z.transpose(1, 2)   # z: (B,H,W,1)
        else:
            kk = d4(k128, g).contiguous(); inv = (lambda z, g=g: d4_inv(z.permute(0, 3, 1, 2), g).permute(0, 2, 3, 1))
        acc = acc + inv(predict(model, data, kk, res, stats))
    return rel_l2_phys(acc / len(group), tgt, stats).cpu().numpy()


# ── training ──────────────────────────────────────────────────────────────────
def build_model(a, res):
    if a.model == "fno":
        assert a.modes <= res // 2, f"modes {a.modes} too large for resolution {res}"
        return get_model("fno", input_channels=3, output_channels=1, width=a.width, modes=a.modes, n_layers=a.layers)
    return get_model("mlp", input_channels=3, output_channels=1, height=res, width=res, hidden_dims=list(a.hidden))


def make_scheduler(opt, a):
    if a.sched == "legacy_step":     # exactly the Stage 2 behaviour: T_max = epochs, stepped once per batch
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.legacy_epochs)
    warm = max(1, int(0.03 * a.steps))
    floor = 1e-3

    def lr_lambda(t):
        if t < warm:
            return (t + 1) / warm
        p = (t - warm) / max(1, a.steps - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    out = (Path(a.out_root) if a.out_root else OUT_ROOT) / a.name
    out.mkdir(parents=True, exist_ok=True)

    data = Data(a.n_train, dev)
    stats = make_stats(data, a.res)
    model = build_model(a, a.res).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = make_scheduler(opt, a)
    mse = nn.MSELoss()

    N = data.k_tr.shape[0]
    spe = max(1, math.ceil(N / a.batch))
    eval_every = a.eval_every or max(20, a.steps // 100)
    best = {"val": float("inf"), "step": -1, "state": None}
    curve = {"step": [], "train_loss": [], "val_rel_l2": [], "lr": []}
    run_loss, n_loss = 0.0, 0
    t0 = time.time()
    step = 0
    perm = torch.randperm(N, generator=gen)
    pos = 0
    while step < a.steps:
        if pos + a.batch > N:                     # new epoch (drop the ragged tail; N >= batch always here)
            perm = torch.randperm(N, generator=gen); pos = 0
        idx = perm[pos:pos + a.batch].to(dev); pos += a.batch
        kb, ub = data.k_tr[idx], data.u_tr[idx]
        if a.aug == "d4":
            t = int(torch.randint(0, 8, (1,), generator=gen))
            kb, ub = d4(kb, t).contiguous(), d4(ub, t).contiguous()
        inp, tgt = data.make(kb, ub, a.res, stats)
        model.train()
        pred = model(inp)
        loss = rel_l2_phys(pred, tgt, stats).mean() if a.loss == "rel_l2" else mse(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step()
        run_loss += float(loss); n_loss += 1
        step += 1
        if step % eval_every == 0 or step == a.steps:
            v = float(evaluate(model, data, data.k_va, data.u_va, a.res, stats).mean())
            curve["step"].append(step); curve["train_loss"].append(run_loss / n_loss)
            curve["val_rel_l2"].append(v); curve["lr"].append(opt.param_groups[0]["lr"])
            run_loss, n_loss = 0.0, 0
            if v < best["val"]:
                best.update(val=v, step=step, state={k: x.detach().clone() for k, x in model.state_dict().items()})
            if not math.isfinite(v):
                break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])

    res_out = {"spec": vars(a), "params": n_params, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
               "train_time_s": train_time, "steps_per_s": step / train_time, "steps_per_epoch": spe,
               "best_step": best["step"], "best_val_rel_l2": best["val"], "stats": stats}
    te = evaluate(model, data, data.k_te, data.u_te, a.res, stats)
    va = evaluate(model, data, data.k_va, data.u_va, a.res, stats)
    res_out.update(test_per_sample=te.tolist(), test_mean=float(te.mean()), test_median=float(np.median(te)),
                   test_std=float(te.std()), test_min=float(te.min()), test_max=float(te.max()),
                   val_mean=float(va.mean()))
    if a.tta:
        tt = evaluate(model, data, data.k_te, data.u_te, a.res, stats, tta=True)
        res_out.update(test_tta_mean=float(tt.mean()), test_tta_per_sample=tt.tolist())
    if a.model == "fno":
        for r in a.eval_res:
            if r != a.res and 128 % r == 0:
                e = evaluate(model, data, data.k_te, data.u_te, r, stats)
                res_out.setdefault("test_at_res", {})[str(r)] = float(e.mean())
                res_out.setdefault("test_at_res_per_sample", {})[str(r)] = e.tolist()
    with open(out / "metrics.json", "w") as f:
        json.dump(res_out, f)
    with open(out / "curve.json", "w") as f:
        json.dump(curve, f)
    if a.save_model:
        torch.save({"model_state_dict": model.state_dict(), "spec": vars(a), "stats": stats}, out / "best_model.pt")
    print(f"[{a.name}] params {n_params/1e6:.2f}M | {step/train_time:.1f} steps/s | val {best['val']:.4f} @ {best['step']} | "
          f"test {res_out['test_mean']:.4f}" + (f" | tta {res_out['test_tta_mean']:.4f}" if a.tta else ""), flush=True)
    return res_out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="fno", choices=["fno", "mlp"])
    p.add_argument("--width", type=int, default=64); p.add_argument("--modes", type=int, default=12)
    p.add_argument("--layers", type=int, default=4); p.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048])
    p.add_argument("--res", type=int, default=64, choices=[32, 64, 128])
    p.add_argument("--n-train", type=int, default=900)
    p.add_argument("--aug", default="none", choices=["none", "d4"])
    p.add_argument("--steps", type=int, default=5700)          # 5700 = the Stage 2 budget (100 epochs x 57 steps)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--loss", default="mse", choices=["mse", "rel_l2"])
    p.add_argument("--sched", default="legacy_step", choices=["legacy_step", "cosine"])
    p.add_argument("--legacy-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-res", type=int, nargs="*", default=[])
    p.add_argument("--tta", action="store_true"); p.add_argument("--save-model", action="store_true")
    p.add_argument("--out-root", default=None, help="override the default results/stage3/runs/ output directory")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
