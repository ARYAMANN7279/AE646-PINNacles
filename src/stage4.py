"""
Stage 4 studies (workshop paper). Same data, split, optimiser, schedule, D4 augmentation and validation-only
checkpoint selection as stage3_train.py, whose functions are reused. New options:

  --arch fno|mlp|unet|deeponet      model (unet / deeponet from stage4_models.py)
  --target-scale none|kappa_h       learn u * kappa_h instead of u, where kappa_h = 1 / mean(1 / kappa) is the map's
                                    harmonic-mean permeability (for uniform kappa, u is exactly proportional to 1/kappa);
                                    predictions are divided by kappa_h again, so the error is always on u itself
  --extra none|hard|random          add training maps from the never-used pool: all "hard" ones (high-permeability
                                    area share < 5 %) or the same number of random ones (control)
  --multires                        each batch at a random grid of 32, 64 or 128 (validation and selection at 64)
  --save-pred                       store the test predictions at 64 x 64 (float16) for ensemble / UQ analysis
  --ood-dir DIR                     also report the error on every out-of-distribution set DIR/*.npz (stage4_ood.py)
  --save-model                      store the best (validation) weights as model.pt
  --train-npz F [--train-mode replace|add]   train on F (kappa128, u128; e.g. data/mixed/train.npz) instead of, or in
                                    addition to, the 900 PDEBench maps;  --val-npz F  validate (select) on F instead
  --log-kappa                       feed log(kappa) to the network instead of kappa
Every run reports the test error at 32, 64 and 128 (not for the MLP, which is tied to its grid) and on the hard
test maps (8 of 200).

Run: python src/stage4.py --name x --arch unet --sched cosine --loss rel_l2 --aug d4 --steps 5700 --seed 0
Output: results/stage4/runs/<name>/metrics.json (+ preds.npz)
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import stage3_train as T
from models import count_parameters
from stage4_models import get_stage4_model

ROOT = T.ROOT
HARD = 0.05
LOGK = False                                                       # --log-kappa: the network sees log(kappa)


def kin(k):
    return torch.log(k) if LOGK else k


def take(argv, flag, default, cast=str, is_bool=False):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if is_bool:
        del argv[i]; return True
    v = cast(argv[i + 1]); del argv[i:i + 2]; return v


def gpu_temp(idx):
    try:
        return int(subprocess.run(["nvidia-smi", "-i", str(idx), "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        return 0


def area_hi(k):
    """High-permeability area share per map (B, H, W) -> (B,)."""
    return (k > 0.5 * (float(k.min()) + float(k.max()))).float().mean(dim=(1, 2))


def kappa_h(k):
    return 1.0 / (1.0 / k).mean(dim=(1, 2))


@torch.no_grad()
def evaluate(model, data, k128, u128, res, stats, scale, return_pred=False):
    model.eval()
    s = 128 // res
    out = T.predict(model, data, kin(k128), res, stats)[..., 0] * stats["u_std"] + stats["u_mean"]
    if scale:
        out = out / kappa_h(k128)[:, None, None]
    tgt = u128[:, ::s, ::s]
    e = ((out - tgt).reshape(len(out), -1).norm(dim=1) / tgt.reshape(len(tgt), -1).norm(dim=1)).cpu().numpy()
    return (e, out) if return_pred else e


def main(argv):
    arch = take(argv, "--arch", "fno")
    scale = take(argv, "--target-scale", "none") == "kappa_h"
    extra = take(argv, "--extra", "none")
    multires = take(argv, "--multires", False, is_bool=True)
    save_pred = take(argv, "--save-pred", False, is_bool=True)
    therm = take(argv, "--throttle-gpu", -1, int)
    ood_dir = take(argv, "--ood-dir", None)
    save_model = take(argv, "--save-model", False, is_bool=True)
    train_npz = take(argv, "--train-npz", None)
    train_mode = take(argv, "--train-mode", "replace")
    val_npz = take(argv, "--val-npz", None)
    global LOGK
    LOGK = take(argv, "--log-kappa", False, is_bool=True)
    if "--out-root" not in argv:
        argv += ["--out-root", str(ROOT / "results" / "stage4" / "runs")]
    if "--model" not in argv:
        argv += ["--model", "mlp" if arch == "mlp" else "fno"]
    a = T.parse(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)

    data = T.Data(900, dev)
    n_extra = 0
    if extra in ("hard", "random"):
        K = np.load(T.CACHE / "kappa128.npy", mmap_mode="r"); U = np.load(T.CACHE / "u128.npy", mmap_mode="r")
        pool = np.load(T.CACHE / "splits.npz")["extra"]
        kmin, kmax = float(data.k_tr.min()), float(data.k_tr.max())
        share = np.array([(K[i][::2, ::2] > 0.5 * (kmin + kmax)).mean() for i in pool])
        hard_idx = pool[share < HARD]
        if extra == "hard":
            pick = hard_idx
        else:
            pick = np.random.default_rng(1000 + a.seed).choice(pool, size=len(hard_idx), replace=False)
        n_extra = len(pick)
        data.k_tr = torch.cat([data.k_tr, torch.tensor(np.stack([K[i] for i in pick]), device=dev)])
        data.u_tr = torch.cat([data.u_tr, torch.tensor(np.stack([U[i] for i in pick]), device=dev)])
    if train_npz:                                                  # external (solver-labelled) training maps
        z = np.load(train_npz)
        kt, ut = torch.tensor(z["kappa128"], device=dev), torch.tensor(z["u128"], device=dev)
        if train_mode == "replace":
            data.k_tr, data.u_tr = kt, ut
        else:
            data.k_tr, data.u_tr = torch.cat([data.k_tr, kt]), torch.cat([data.u_tr, ut])
        n_extra = len(kt)
    if val_npz:
        z = np.load(val_npz)
        data.k_va, data.u_va = torch.tensor(z["kappa128"], device=dev), torch.tensor(z["u128"], device=dev)
    if scale:
        data.u_tr = data.u_tr * kappa_h(data.k_tr)[:, None, None]
    import types
    stats = T.make_stats(types.SimpleNamespace(k_tr=kin(data.k_tr), u_tr=data.u_tr), a.res)

    if arch in ("fno", "mlp"):
        model = T.build_model(a, a.res).to(dev)
    else:
        model = get_stage4_model(arch).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = T.make_scheduler(opt, a)

    N = data.k_tr.shape[0]
    eval_every = a.eval_every or max(20, a.steps // 100)
    best = {"val": float("inf"), "step": -1, "state": None}
    curve = {"step": [], "val": []}
    perm, pos, step, t0 = torch.randperm(N, generator=gen), 0, 0, time.time()
    resg = [32, 64, 128]
    while step < a.steps:
        if pos + a.batch > N:
            perm, pos = torch.randperm(N, generator=gen), 0
        idx = perm[pos:pos + a.batch].to(dev); pos += a.batch
        kb, ub = data.k_tr[idx], data.u_tr[idx]
        if a.aug == "d4":
            t = int(torch.randint(0, 8, (1,), generator=gen))
            kb, ub = T.d4(kb, t).contiguous(), T.d4(ub, t).contiguous()
        r = resg[int(torch.randint(0, 3, (1,), generator=gen))] if multires else a.res
        inp, tgt = data.make(kin(kb), ub, r, stats)
        model.train()
        loss = T.rel_l2_phys(model(inp), tgt, stats).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        step += 1
        if therm >= 0 and step % 500 == 0:
            while gpu_temp(therm) >= 86:
                time.sleep(5)
        if step % eval_every == 0 or step == a.steps:
            v = float(evaluate(model, data, data.k_va, data.u_va, a.res, stats, scale).mean())
            curve["step"].append(step); curve["val"].append(v)
            if v < best["val"]:
                best.update(val=v, step=step, state={k: x.detach().clone() for k, x in model.state_dict().items()})
            if not math.isfinite(v):
                break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    te, pred = evaluate(model, data, data.k_te, data.u_te, a.res, stats, scale, return_pred=True)
    hard = (area_hi(data.k_te[:, ::2, ::2]) < HARD).cpu().numpy()
    res = {"spec": vars(a), "arch": arch, "target_scale": "kappa_h" if scale else "none", "extra": extra, "n_extra": n_extra,
           "multires": multires, "train_npz": Path(train_npz).name if train_npz else None, "train_mode": train_mode if train_npz else None,
           "val_npz": Path(val_npz).name if val_npz else None, "log_kappa": LOGK, "params": n_params, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "train_time_s": train_time, "best_step": best["step"], "best_val": best["val"],
           "test_mean": float(te.mean()), "test_median": float(np.median(te)), "test_per_sample": te.tolist(),
           "n_hard_test": int(hard.sum()), "test_hard": float(te[hard].mean()), "test_rest": float(te[~hard].mean())}
    if arch != "mlp":
        res["test_at_res"] = {str(r): float(evaluate(model, data, data.k_te, data.u_te, r, stats, scale).mean()) for r in (32, 64, 128)}
    if ood_dir:                                                    # out-of-distribution sets (stage4_ood.py), error at 64 x 64
        res["ood"] = {}
        for f in sorted(Path(ood_dir).glob("*.npz")):
            z = np.load(f)
            e = evaluate(model, data, torch.tensor(z["kappa128"], device=dev), torch.tensor(z["u128"], device=dev), a.res, stats, scale)
            res["ood"][f.stem] = {"mean": float(e.mean()), "median": float(np.median(e)), "per_sample": e.tolist()}
    if save_model:
        torch.save({"state": model.state_dict(), "arch": arch, "spec": vars(a), "stats": stats, "scale": scale}, out / "model.pt")
    json.dump(res, open(out / "metrics.json", "w"))
    json.dump(curve, open(out / "curve.json", "w"))
    if save_pred:
        np.savez_compressed(out / "preds.npz", pred=pred.cpu().numpy().astype(np.float16))
    print(f"[{a.name}] {arch} params {n_params/1e6:.2f}M | val {best['val']:.4f} | test {res['test_mean']:.4f} | hard {res['test_hard']:.3f}"
          + (f" | res32/64/128 {res['test_at_res']['32']:.3f}/{res['test_at_res']['64']:.3f}/{res['test_at_res']['128']:.3f}" if arch != "mlp" else "")
          + f" | {train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
