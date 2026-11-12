"""
Stage 3 extension studies (added after the final presentation, on the professor's suggestions).

Same data, models, optimiser, schedule, D4 augmentation, selection protocol and evaluation as stage3_train.py
(this file reuses its functions; only the training loop is re-written because the weights need the sample index).

  --noise s     NOISY PRESSURE LABELS. Gaussian noise is added once (fixed, seeded) to the TRAINING and
                VALIDATION pressure maps: u_noisy = u + s * rms(u_i) * N(0,1), i.e. the noise RMS is a fraction s of
                each map's own RMS (s = 0.05 -> 5 %). Checkpoints are selected on the noisy validation maps (all that a
                practitioner would have). The TEST maps stay clean, so the reported error is the error against the truth.
  --adaptive a  SELF-ADJUSTING PER-SAMPLE WEIGHTS. Every training map i keeps a running (EMA) estimate e_i of its own
                loss. The loss weight is w_i = clip( (e_i / mean e)^a , w_min, w_max ), renormalised to mean 1, so
                maps that are currently fitted badly (e.g. nearly uniform permeability) count more. a = 0 is the usual loss.

Run: python src/stage3_robust.py --name x --model fno --layers 4 --sched cosine --loss rel_l2 --aug d4 --tta --steps 5700 \
         --seed 0 --noise 0.05 --adaptive 1.0
Output: results/stage3_extra/runs/<name>/metrics.json (per-sample test errors, test-map high-permeability share, weights).
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

ROOT = T.ROOT


def take(argv, flag, default, n=1, cast=float):
    """Remove `flag v1..vn` from argv and return the value(s)."""
    if flag not in argv:
        return default
    i = argv.index(flag)
    vals = [cast(v) for v in argv[i + 1:i + 1 + n]]
    del argv[i:i + 1 + n]
    return vals[0] if n == 1 else vals


def gpu_temp(idx):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(idx), "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip())
    except Exception:
        return 0


def main(argv):
    noise = take(argv, "--noise", 0.0)
    alpha = take(argv, "--adaptive", 0.0)
    w_min, w_max = take(argv, "--w-clip", [0.2, 5.0], n=2)
    ema_m = take(argv, "--w-ema", 0.9)
    therm = take(argv, "--throttle-gpu", -1, cast=int)      # physical GPU index to watch (shared, cooling-limited box)
    if "--out-root" not in argv:
        argv += ["--out-root", str(ROOT / "results" / "stage3_extra" / "runs")]
    a = T.parse(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)

    data = T.Data(a.n_train, dev)
    u_va_clean = data.u_va.clone()
    if noise > 0:
        g = torch.Generator(device="cpu").manual_seed(10_000 + a.seed)

        def noisy(u):
            rms = u.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            return u + noise * rms * torch.randn(u.shape, generator=g).to(u.device)
        data.u_tr = noisy(data.u_tr)
        data.u_va = noisy(data.u_va)
    stats = T.make_stats(data, a.res)
    model = T.build_model(a, a.res).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = T.make_scheduler(opt, a)

    N = data.k_tr.shape[0]
    eval_every = a.eval_every or max(20, a.steps // 100)
    best = {"val": float("inf"), "step": -1, "state": None}
    w = torch.ones(N, device=dev)
    ema = torch.zeros(N, device=dev)
    seen = torch.zeros(N, dtype=torch.bool, device=dev)
    curve = {"step": [], "train_loss": [], "val_rel_l2": []}
    run_loss, n_loss, step, t0 = 0.0, 0, 0, time.time()
    perm, pos = torch.randperm(N, generator=gen), 0
    while step < a.steps:
        if pos + a.batch > N:
            perm, pos = torch.randperm(N, generator=gen), 0
        idx = perm[pos:pos + a.batch].to(dev); pos += a.batch
        kb, ub = data.k_tr[idx], data.u_tr[idx]
        if a.aug == "d4":
            t = int(torch.randint(0, 8, (1,), generator=gen))
            kb, ub = T.d4(kb, t).contiguous(), T.d4(ub, t).contiguous()
        inp, tgt = data.make(kb, ub, a.res, stats)
        model.train()
        pred = model(inp)
        per = T.rel_l2_phys(pred, tgt, stats) if a.loss == "rel_l2" else (pred - tgt).pow(2).mean(dim=(1, 2, 3))
        loss = (w[idx] * per).mean() if alpha > 0 else per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if alpha > 0:
            with torch.no_grad():
                d = per.detach()
                ema[idx] = torch.where(seen[idx], ema_m * ema[idx] + (1 - ema_m) * d, d)
                seen[idx] = True
                e = ema.clone()
                e[~seen] = e[seen].mean()
                w = (e / e.mean()).pow(alpha).clamp(w_min, w_max)
                w = w / w.mean()
        run_loss += float(loss); n_loss += 1; step += 1
        if therm >= 0 and step % 500 == 0:
            while gpu_temp(therm) >= 86:                      # duty-cycle throttle: pause until the card cools
                time.sleep(5)
        if step % eval_every == 0 or step == a.steps:
            v = float(T.evaluate(model, data, data.k_va, data.u_va, a.res, stats).mean())
            curve["step"].append(step); curve["train_loss"].append(run_loss / n_loss); curve["val_rel_l2"].append(v)
            run_loss, n_loss = 0.0, 0
            if v < best["val"]:
                best.update(val=v, step=step, state={k: x.detach().clone() for k, x in model.state_dict().items()})
            if not math.isfinite(v):
                break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    te = T.evaluate(model, data, data.k_te, data.u_te, a.res, stats)
    va = T.evaluate(model, data, data.k_va, u_va_clean, a.res, stats)
    s = 128 // a.res
    area = (data.k_te[:, ::s, ::s] > 0.5 * (float(data.k_te.min()) + float(data.k_te.max()))).float().mean(dim=(1, 2))
    res = {"spec": vars(a), "noise": noise, "adaptive": alpha, "w_clip": [w_min, w_max], "params": n_params,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "train_time_s": train_time,
           "best_step": best["step"], "best_val_selection": best["val"], "val_clean_mean": float(va.mean()),
           "test_mean": float(te.mean()), "test_median": float(np.median(te)), "test_per_sample": te.tolist(),
           "test_area_hi": area.cpu().tolist(), "final_weights": w.cpu().tolist() if alpha > 0 else None}
    if a.tta:
        tt = T.evaluate(model, data, data.k_te, data.u_te, a.res, stats, tta=True)
        res.update(test_tta_mean=float(tt.mean()), test_tta_per_sample=tt.tolist())
    json.dump(res, open(out / "metrics.json", "w"))
    json.dump(curve, open(out / "curve.json", "w"))
    print(f"[{a.name}] noise {noise} adaptive {alpha} | val(sel) {best['val']:.4f} | test {res['test_mean']:.4f}"
          + (f" | tta {res['test_tta_mean']:.4f}" if a.tta else "") + f" | {train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
