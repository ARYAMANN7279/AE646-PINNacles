"""
Real preprocessing ablation on the MLP baseline.

Rebuilds the exact same 900/100/200 split as src/preprocess.py (same seed, same
indices, same stride-2 64x64 subsampling, same targets) and varies ONE
preprocessing choice at a time, retraining the 3x2048 AdamW MLP from scratch
for several seeds each:

  reference        : inputs [kappa, x, y], standardised inputs and targets
  no-coords        : inputs [kappa] only (no x/y coordinate channels)
  no-input-norm    : kappa left in raw physical units (0.1 / 1.0)
  no-target-norm   : pressure left in raw physical units

Because the target field is identical in every variant, the test metric
(relative L2 in physical units, 200 held-out samples) is directly comparable
across variants. Every number is a real training run; nothing is hand-picked.

Run: python src/ablation_preprocess.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import preprocess as pp
from models import get_model
from train import set_seed, train_epoch, evaluate

VARIANTS = [
    ("reference (standardised, +coords)", dict(coords=True,  norm_in=True,  norm_out=True)),
    ("no coordinate channels",            dict(coords=False, norm_in=True,  norm_out=True)),
    ("no input normalisation",            dict(coords=True,  norm_in=False, norm_out=True)),
    ("no target normalisation",           dict(coords=True,  norm_in=True,  norm_out=False)),
]


def build_arrays():
    """Same subset / split / downsampling as src/preprocess.py (physical units)."""
    nu, tensor, x, y = pp.load_raw(pp.RAW_FILE)
    rng = np.random.default_rng(pp.SEED)
    perm = rng.permutation(nu.shape[0])
    trainval_idx = perm[:pp.N_TRAINVAL]
    test_idx = perm[pp.N_TRAINVAL:pp.N_TRAINVAL + pp.N_TEST]
    nu_tv, u_tv = pp.downsample(nu[trainval_idx]), pp.downsample(tensor[trainval_idx])
    nu_te, u_te = pp.downsample(nu[test_idx]), pp.downsample(tensor[test_idx])
    split_perm = np.random.default_rng(pp.SEED).permutation(pp.N_TRAINVAL)
    tr, va = split_perm[:900], split_perm[900:]
    return (nu_tv[tr], u_tv[tr]), (nu_tv[va], u_tv[va]), (nu_te, u_te), x[::2], y[::2]


def make_loaders(cfg, train, val, test, x, y, batch_size, seed):
    c_tr, u_tr = train
    c_mean, c_std = (c_tr.mean(), c_tr.std()) if cfg["norm_in"] else (0.0, 1.0)
    u_mean, u_std = (u_tr.mean(), u_tr.std()) if cfg["norm_out"] else (0.0, 1.0)

    def prep(c, u):
        c = (c - c_mean) / c_std
        u = (u - u_mean) / u_std
        if cfg["coords"]:
            inp, tgt = pp.add_coordinates(c, u, x, y)
        else:
            inp = c[..., None].astype(np.float32)
            tgt = u[..., None].astype(np.float32)
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()

    ds = [TensorDataset(*prep(*s)) for s in (train, val, test)]
    gen = torch.Generator().manual_seed(seed)
    loaders = (
        DataLoader(ds[0], batch_size=batch_size, shuffle=True, generator=gen),
        DataLoader(ds[1], batch_size=batch_size),
        DataLoader(ds[2], batch_size=batch_size),
    )
    in_ch = 3 if cfg["coords"] else 1
    return loaders, float(u_mean), float(u_std), in_ch


def run_one(cfg, data, seed, epochs, batch_size, device):
    set_seed(seed)
    (tr_l, va_l, te_l), u_mean, u_std, in_ch = make_loaders(cfg, *data, batch_size, seed)
    model = get_model("mlp", input_channels=in_ch, output_channels=1,
                      height=64, width=64, hidden_dims=[2048, 2048, 2048]).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        train_epoch(model, tr_l, opt, crit, device, u_mean, u_std, sched)
        v = evaluate(model, va_l, crit, device, u_mean, u_std)["rel_l2"]
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    s = evaluate(model, te_l, crit, device, u_mean, u_std)["sample_rel_l2"]
    return float(s.mean()), float(np.median(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--output", default="results/ablation_preprocess.json")
    ap.add_argument("--figure", default="results/figures/fig10_preprocess_ablation.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    data = build_arrays()
    print("split sizes:", [len(d[0]) for d in data[:3]])

    results = []
    for name, cfg in VARIANTS:
        means, medians = [], []
        for seed in args.seeds:
            m, med = run_one(cfg, data, seed, args.epochs, args.batch_size, device)
            means.append(m); medians.append(med)
            print(f"  {name:38s} seed {seed}: mean rel L2 = {m:.4f}")
        results.append({
            "name": name, **cfg, "seeds": args.seeds,
            "test_mean_rel_l2_per_seed": means,
            "test_mean_rel_l2_avg": float(np.mean(means)),
            "test_mean_rel_l2_std_over_seeds": float(np.std(means)),
            "test_median_rel_l2_avg": float(np.mean(medians)),
        })
        print(f"==> {name}: {np.mean(means):.4f} +/- {np.std(means):.4f} (over {len(means)} seeds)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    names = [r["name"] for r in results]
    avg = [r["test_mean_rel_l2_avg"] for r in results]
    sd = [r["test_mean_rel_l2_std_over_seeds"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), avg, yerr=sd, capsize=5,
                  color=["#DD8452"] + ["#4C72B0"] * (len(names) - 1))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Test mean relative $L_2$ error")
    ax.set_title(f"MLP preprocessing ablation (mean $\\pm$ std over {len(args.seeds)} seeds)")
    for b, m in zip(bars, avg):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.003, f"{m:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    fig.savefig(str(args.figure).replace(".png", ".pdf"))
    print(f"Saved -> {args.output}, {args.figure}")


if __name__ == "__main__":
    main()
