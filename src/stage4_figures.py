"""Stage 4 figures from results/stage4/summary.json (+ preds.npz) -> results/stage4/figures/*.{png,pdf}.
Run from the repo root after src/stage4_aggregate.py:  python src/stage4_figures.py"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R4 = ROOT / "results" / "stage4"
OUT = R4 / "figures"; OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, GREEN, ORANGE, GREY, RED = "#142F4E", "#2F7FD1", "#1E8E5A", "#C96F1E", "#5F6B7A", "#B03A2E"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open(R4 / "summary.json"))


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=200); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)


# 1. error on each grid, per model (trained at 64 x 64)
cfgs = [("s4_fno_ref", "FNO", GREEN), ("s4_fno_multires", "FNO, multi-res training", BLUE), ("s4_unet", "U-Net", ORANGE), ("s4_deeponet", "DeepONet", GREY)]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
x = np.arange(3); w = 0.2
for i, (c, lab, col) in enumerate(cfgs):
    if c not in S: continue
    m = [S[c]["at_res"][r]["mean"] for r in ("32", "64", "128")]; sd = [S[c]["at_res"][r]["sd"] for r in ("32", "64", "128")]
    ax.bar(x + (i - 1.5) * w, m, w, yerr=sd, color=col, label=lab, capsize=2)
ax.set_xticks(x); ax.set_xticklabels(["32 x 32", "64 x 64 (training grid)", "128 x 128"]); ax.set_yscale("log")
ax.set_ylabel("test rel. L2 error"); ax.legend(fontsize=8.5, frameon=False); ax.set_title("Same accuracy on the training grid, very different transfer", fontsize=10, color=NAVY)
save(fig, "fig_s4_resolution")

# 2. hard maps: what helps
rows = [("s4_fno_ref", "baseline"), ("s4_fno_scale", "target / kappa_h"), ("s4_fno_xrand", "+230 random maps"),
        ("s4_fno_xhard", "+230 hard maps"), ("s4_fno_scale_xhard", "+230 hard, scaled")]
ex = ROOT / "results" / "stage3_extra" / "summary.json"
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
labs = [l for c, l in rows if c in S]; hard = [S[c]["hard"] for c, _ in rows if c in S]; rest = [S[c]["rest"] for c, _ in rows if c in S]
if ex.exists():
    a1 = json.load(open(ex))["adaptive"]["alpha1"]
    labs.insert(2, "adaptive weights"); hard.insert(2, a1["hard"]); rest.insert(2, a1["rest"])
cols = [GREY] * len(labs); cols[labs.index("+230 hard maps")] = GREEN
for ax, vals, t in ((axs[0], hard, "8 hard test maps"), (axs[1], rest, "192 other test maps")):
    ax.barh(range(len(labs)), [v["mean"] for v in vals], xerr=[v["sd"] for v in vals], color=cols, capsize=2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs if ax is axs[0] else [""] * len(labs), fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("test rel. L2 error"); ax.set_title(t, fontsize=10, color=NAVY)
save(fig, "fig_s4_hard")

# 3. ensemble spread vs error (FNO, 3 seeds)
truth = np.load(ROOT / "results" / "stage3" / "diagnostics_arrays.npz")["gt"].astype(np.float64)
files = sorted((R4 / "runs").glob("s4_fno_ref_s*/preds.npz"))
if files:
    P = np.stack([np.load(f)["pred"].astype(np.float64) for f in files]); tn = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    err = np.linalg.norm((P.mean(0) - truth).reshape(len(truth), -1), axis=1) / tn
    spr = np.linalg.norm(P.std(0).reshape(len(truth), -1), axis=1) / tn
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(spr, err, s=12, color=BLUE, alpha=.8); ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("ensemble spread (3 seeds)"); ax.set_ylabel("error of the ensemble mean")
    rho = S["s4_fno_ref"]["ensemble"]["spearman_spread_vs_error"]
    ax.set_title(f"Spread flags the bad maps (Spearman {rho:.2f})", fontsize=10, color=NAVY)
    save(fig, "fig_s4_uq")
print(sorted(p.name for p in OUT.glob("*.png")))

# 4. out-of-distribution heat map (configurations evaluated with --ood-dir)
sets = ["ID", "rough", "smooth", "mostly_low", "contrast_lo", "three_phase", "contrast_hi"]
setlab = ["same stats", "rougher", "smoother", "mostly low", "contrast 0.3", "three phases", "contrast 0.01"]
names = {"s5_fno_pde": "FNO, PDEBench", "s5_unet_pde": "U-Net, PDEBench", "s5_deeponet_pde": "DeepONet, PDEBench",
         "s5_fno_mixed": "FNO, random contrast", "s5_fno_mixed_logk": "FNO, random contrast, log k",
         "s5_unet_mixed_logk": "U-Net, random contrast, log k", "s5_fno_pdemix_logk": "FNO, PDEBench + random, log k",
         "s7_fno_pdemixwide_logk": "FNO, PDEBench + wide random, log k", "s7_fno_all": "FNO, all fixes",
         "s7_fno_allwide": "FNO, all fixes, wide range", "s8_fno_final_robust": "FNO final, PDEBench + wide random, log k"}
# the Stage 3 final FNO (PDEBench only, 8 layers, 60k steps): only seed 0's checkpoint survived, evaluated by hand
_f = R4 / "ood_final_fno_stage3_s0.json"
if _f.exists():
    _o = json.load(open(_f))
    S["stage3_final_fno"] = {"test": {"mean": 0.0202}, "ood": {k: {"mean": v[0]} for k, v in _o.items()}}
    names = {"stage3_final_fno": "FNO final, PDEBench (1 checkpoint)", **names}
rows = [c for c in names if c in S and "ood" in S[c]]
if rows:
    M = np.array([[S[c]["test"]["mean"]] + [S[c]["ood"][s]["mean"] for s in sets] for c in rows])
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(rows) + 1.4))
    im = ax.imshow(np.log10(M), cmap="RdYlGn_r", vmin=-2.3, vmax=0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks(range(len(sets) + 1)); ax.set_xticklabels(["PDEBench test"] + setlab, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[c] for c in rows], fontsize=8.5); ax.grid(False)
    ax.set_title("Test error in and out of distribution (mean of 3 seeds)", fontsize=10, color=NAVY)
    save(fig, "fig_s4_ood")
print("ood figure:", (OUT / "fig_s4_ood.png").exists())
