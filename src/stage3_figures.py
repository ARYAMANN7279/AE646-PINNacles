"""Stage 3 figures from results/stage3/*.json -> stage3/figures/*.{png,pdf}.  Run: python src/stage3_figures.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage3_aggregate import load

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "stage3"
OUT = ROOT / "stage3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#1f4e79", "#ca6f1e", "#1e8b4c", "#b03a2e", "#7f8c8d"
rows = load()


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_recipe():
    g = defaultdict(list)
    for n, r in rows.items():
        m = re.match(r"e1_(\w+?)_(leg|cos)_(mse|rel_l2)_(none|d4)_s\d", n)
        if m:
            g[m.groups()].append(r)
    order = [("leg", "mse", "none", "Stage 2 recipe"), ("cos", "mse", "none", "+ single cosine"), ("leg", "rel_l2", "none", "+ rel-L2 loss (only)"),
             ("leg", "mse", "d4", "+ D4 aug (only)"), ("cos", "rel_l2", "none", "cosine + rel-L2"), ("cos", "rel_l2", "d4", "full recipe")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(order)); w = 0.38
    for j, (model, c) in enumerate((("fno", GREEN), ("mlp", ORANGE))):
        m = [np.mean([r["test"] for r in g[(model,) + o[:3]]]) for o in order]
        s = [np.std([r["test"] for r in g[(model,) + o[:3]]], ddof=1) for o in order]
        t = [np.mean([r["tta"] for r in g[(model,) + o[:3]]]) for o in order]
        ax.bar(x + (j - .5) * w, m, w, yerr=s, color=c, label=model.upper(), capsize=2)
        ax.plot(x + (j - .5) * w, t, "k_", ms=10, mew=2, label="with D4 TTA" if j == 0 else None)
        for xi, v in zip(x + (j - .5) * w, m):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([o[3] for o in order], rotation=18, ha="right"); ax.set_ylabel("test rel. L2 (mean of 3 seeds)")
    ax.legend(); save(fig, "fig_recipe_ablation")


def sweep(prefix, key, label, ax, cast=float):
    pts = sorted((cast(re.sub(prefix, "", n)), r) for n, r in rows.items() if re.fullmatch(prefix + r"[0-9.e\-]+", n))
    xs = [p[0] for p in pts]
    ax.plot(xs, [p[1]["val"] for p in pts], "o-", color=BLUE, label="val")
    ax.plot(xs, [p[1]["test"] for p in pts], "s--", color=GREEN, label="test")
    ax.set_xlabel(label); ax.set_ylim(0.018, 0.038)


def fig_arch():
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    sweep("e2_modes", "modes", "Fourier modes / layer", axs[0], int); sweep("e2_width", "w", "width", axs[1], int)
    sweep("e2_layers", "l", "layers", axs[2], int); sweep("e2_steps", "s", "training steps", axs[3], int); axs[3].set_xscale("log"); axs[3].set_xticks([5700, 20000, 80000]); axs[3].set_xticklabels(["5.7k", "20k", "80k"]); axs[3].minorticks_off()
    axs[0].set_ylabel("rel. L2"); axs[0].legend(); fig.tight_layout(); save(fig, "fig_arch_sweeps")


def fig_scaling():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for model, c in (("fno", GREEN), ("mlp", ORANGE)):
        d = defaultdict(list)
        for n, r in rows.items():
            m = re.match(rf"e3_{model}_n(\d+)_s\d", n)
            if m:
                d[int(m.group(1))].append(r["tta"])
        ns = sorted(d); ax.errorbar(ns, [np.mean(d[n]) for n in ns], [np.std(d[n]) for n in ns], fmt="o-", color=c, label=model.upper(), capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test rel. L2 (D4 TTA)"); ax.legend()
    ax.axvline(900, color=GREY, ls=":"); ax.text(950, ax.get_ylim()[1] * 0.9, "course split (900)", fontsize=7, color=GREY); save(fig, "fig_data_scaling")


def fig_resolution():
    M = np.full((3, 3), np.nan); rs = [32, 64, 128]
    for i, tr in enumerate(rs):
        for j, te in enumerate(rs):
            v = [r["test"] if te == tr else (r["at_res"] or {}).get(str(te)) for n, r in rows.items() if re.fullmatch(rf"e4_fno_res{tr}_s\d", n)]
            M[i, j] = np.mean(v)
    fig, ax = plt.subplots(figsize=(4.2, 3.4)); im = ax.imshow(M, cmap="YlOrRd")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold" if i == j else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(rs); ax.set_yticks(range(3)); ax.set_yticklabels(rs); ax.grid(False)
    ax.set_xlabel("evaluation resolution"); ax.set_ylabel("training resolution"); save(fig, "fig_resolution")


def fig_curves():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for n, lab, c in (("e1_fno_leg_mse_none_s0", "Stage 2 recipe", RED), ("e1_fno_cos_rel_l2_d4_s0", "full recipe (5,700 steps)", GREEN), ("final_fno_s0", "final FNO (60,000 steps)", BLUE)):
        p = R / "runs" / n / "curve.json"
        if p.exists():
            d = json.load(open(p)); ax.plot(np.array(d["step"]), d["val_rel_l2"], color=c, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("validation rel. L2"); ax.legend(); save(fig, "fig_curves")


def fig_pareto():
    p = R / "pareto.json"
    if not p.exists():
        return
    d = json.load(open(p)); fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for r in d:
        ax.scatter(r["ms"], r["err"], color=r["color"], marker=r["marker"], s=50, label=r["label"])
        ax.annotate(r["note"], (r["ms"], r["err"]), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("time per solution [ms] (amortised, batched where stated)"); ax.set_ylabel("rel. L2 error vs PDEBench truth")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7); save(fig, "fig_pareto")


def fig_diag():
    p = R / "diagnostics_arrays.npz"
    if not p.exists():
        return
    d = np.load(p); dj = json.load(open(R / "diagnostics.json"))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    ax = axs[0]; ax.scatter(d["area_hi"], d["rel"], s=10, color=BLUE); ax.set_xlabel("high-κ area fraction"); ax.set_ylabel("rel. L2 error"); ax.set_title("error vs κ composition", fontsize=9)
    ax = axs[1]; ie = dj["interface_error"]; ax.bar(range(len(ie)), [b["mean_abs_err_over_peak"] for b in ie], color=GREEN)
    ax.set_xticks(range(len(ie))); ax.set_xticklabels(["0-1", "1-2", "2-4", "4-8", "8-16", ">16"]); ax.set_xlabel("distance to κ interface [cells]"); ax.set_ylabel("mean |error| / peak u"); ax.set_title("error vs interface distance", fontsize=9)
    ax = axs[2]; fl = dj["spectral_floor"]; ms = sorted(int(k) for k in fl)
    ax.semilogy(ms, [fl[str(m)] for m in ms], "o-", color=RED, label="truncation floor of the truth")
    mod = sorted((int(re.sub("e2_modes", "", n)), r["test"]) for n, r in rows.items() if re.fullmatch(r"e2_modes\d+", n))
    ax.semilogy([a for a, b in mod], [b for a, b in mod], "s--", color=GREEN, label="trained FNO test error")
    ax.set_xlabel("retained Fourier modes"); ax.set_ylabel("rel. L2"); ax.legend(fontsize=7); ax.set_title("spectral truncation", fontsize=9)
    fig.tight_layout(); save(fig, "fig_diagnostics")
    # sample maps: best / median / worst
    rel = d["rel"]; idx = [int(np.argmin(rel)), int(np.argsort(rel)[len(rel) // 2]), int(np.argmax(rel))]
    fig, axs = plt.subplots(3, 4, figsize=(9, 6.6))
    for r_, i in enumerate(idx):
        vm = d["gt"][i].max()
        for c, (arr, t, cm, kw) in enumerate(((d["kappa"][i], "κ", "gray", {}), (d["gt"][i], "truth", "viridis", dict(vmin=0, vmax=vm)),
                                              (d["pred"][i], "FNO", "viridis", dict(vmin=0, vmax=vm)), (np.abs(d["pred"][i] - d["gt"][i]), "|error|", "magma", {}))):
            im = axs[r_, c].imshow(arr, cmap=cm, origin="lower", **kw); axs[r_, c].set_xticks([]); axs[r_, c].grid(False)
            axs[r_, c].set_title(f"{t}" + (f" (rel L2 {rel[i]:.3f})" if c == 0 else ""), fontsize=8)
            fig.colorbar(im, ax=axs[r_, c], fraction=0.046)
    fig.tight_layout(); save(fig, "fig_samples")


if __name__ == "__main__":
    for f in (fig_recipe, fig_arch, fig_scaling, fig_resolution, fig_curves, fig_pareto, fig_diag):
        f(); print("ok", f.__name__)
