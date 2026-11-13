"""Re-render the deck and robustness figures as vector PDFs for the A0 poster (writes only into poster/figs).
Run from the repo root: python3 poster/make_figs.py"""
import os, runpy, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "poster" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
_orig = Figure.savefig

def redirected(self, fname, *a, **k):            # every figure goes to poster/figs as PDF (and PNG), nowhere else
    stem = Path(str(fname)).stem
    k.pop("dpi", None)
    _orig(self, OUT / f"{stem}.pdf", *a, **k)
    _orig(self, OUT / f"{stem}.png", *a, dpi=300, **k)
Figure.savefig = redirected

os.chdir(OUT)
runpy.run_path(str(ROOT / "stage3" / "deck_extension" / "make_figures.py"), run_name="__main__")
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "stage3_robust_aggregate.py"), run_name="__main__")
print(sorted(p.name for p in OUT.glob("*.pdf")))

# poster-shaped physics figure (same data as the deck's fig_deck_physics, larger labels)
import numpy as np
import matplotlib.pyplot as plt
D = np.load(ROOT / "results" / "stage3" / "diagnostics_arrays.npz"); rel = D["rel"]; area = D["area_hi"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
ax = axs[0]
ax.hist(rel, bins=np.linspace(0, 0.4, 21), color="#1E8E5A", edgecolor="white")
ax.axvline(np.median(rel), color="#142F4E", ls="--", lw=1.5)
ax.text(0.05, ax.get_ylim()[1] * 0.6, f"typical map\n{np.median(rel):.3f}", fontsize=12, color="#142F4E")
ax.set_xlabel("error of one test map", fontsize=12); ax.set_ylabel("number of maps", fontsize=12)
ax.set_title(f"{(rel < 0.05).mean() * 100:.0f}% of maps under 5% error", fontsize=12.5, color="#142F4E")
ax = axs[1]
ax.scatter(area, rel, s=16, color="#2F7FD1", alpha=.8)
ax.set_xlabel("share of map with high permeability", fontsize=12); ax.set_ylabel("error", fontsize=12)
ax.set_title("worst maps: almost all low permeability", fontsize=12.5, color="#142F4E")
for a_ in axs:
    a_.tick_params(labelsize=11)
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("fig_poster_physics.png")

# ------------------------------------------------------------------ poster-sized redraws (labels about 14-18 pt at print size)
import json, re
from collections import defaultdict
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.text as mtext
sys.path.insert(0, str(ROOT / "src"))
import stage3_figures as F                                   # reuses its data loading and sweep helper
plt.rcParams.update({"font.size": 14, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
NAVY, BLUE, GREEN, ORANGE, GREY = "#142F4E", "#2F7FD1", "#1E8E5A", "#C96F1E", "#5F6B7A"
COLW = 9.45                                                   # poster column text width in inches

def floor_fonts(fig, mn):
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < mn:
            t.set_fontsize(mn)

# model diagram
fig, axs = plt.subplots(2, 1, figsize=(COLW, 4.6))
def box(ax, x, w, txt, fc, ec, y=0.15, h=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=13, color=NAVY)
def arr(ax, x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=16, color=GREY, lw=1.5))
for ax in axs:
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.15); ax.axis("off")
ax = axs[0]; ax.text(0, 1.05, "MLP  (100.7 M parameters)", fontsize=15, fontweight="bold", color=ORANGE, va="center")
xs = [0, 2.05, 4.1, 6.15, 8.2]; ws = 1.8
labels = ["Input\n12,288\nnumbers", "Hidden 1\n4,096\n+ GELU", "Hidden 2\n4,096\n+ GELU", "Hidden 3\n4,096\n+ GELU", "Output\n4,096 =\n64 x 64"]
for i, (x, t) in enumerate(zip(xs, labels)):
    box(ax, x, ws, t, "#F3F7FC" if i in (0, 4) else "#FFF6EC", BLUE if i in (0, 4) else ORANGE)
    if i < 4: arr(ax, x + ws, xs[i + 1])
ax = axs[1]; ax.text(0, 1.05, "FNO  (9.5 M parameters)", fontsize=15, fontweight="bold", color=GREEN, va="center")
spec = [(0, 1.45, "Input\nκ, x, y"), (1.75, 1.35, "Lift\n3 → 64"), (3.4, 3.3, "8 x Fourier layer\nFFT, keep 12 x 12 modes,\nweights, inverse FFT,\n+ 1x1 skip, GELU"),
        (7.0, 1.45, "Project\n64→128→1"), (8.75, 1.25, "Output\n64 x 64")]
for i, (x, w, t) in enumerate(spec):
    box(ax, x, w, t, "#F3F7FC" if i in (0, 4) else "#ECF6EF", BLUE if i in (0, 4) else GREEN, y=0.05, h=0.9)
    if i < 4: arr(ax, x + w, spec[i + 1][0])
fig.tight_layout(); fig.savefig("fig_poster_models.png")

# solver vs FNO
P = json.load(open(ROOT / "results" / "stage3" / "pareto.json"))
sol = [p for p in P if p["label"].startswith("FV solver, direct")]; fno = [p for p in P if p["label"] == "FNO, GPU"]
fig, ax = plt.subplots(figsize=(COLW * 0.92, 5.2))
ax.plot([p["ms"] for p in sol], [p["err"] for p in sol], "-o", color=GREY, ms=11, lw=2.2, label="finite-volume solver (1 CPU thread)")
for p in sol:
    n = p["note"].replace("N=", ""); ax.annotate(f"{n} x {n}", (p["ms"], p["err"]), textcoords="offset points", xytext=(10, 8), fontsize=14, color=GREY)
ax.plot([p["ms"] for p in fno], [p["err"] for p in fno], "s", color=GREEN, ms=13, label="FNO on a GPU")
for p in fno:
    one = p["note"] == "batch 1"
    ax.annotate("one map at a time" if one else "128 maps at once", (p["ms"], p["err"]), textcoords="offset points",
                xytext=(0, 16) if one else (0, -26), ha="center", fontsize=14, color=GREEN)
ax.set_xscale("log"); ax.set_xlim(0.08, 120); ax.set_ylim(0, 0.11)
ax.set_xlabel("time per solution (ms, log scale)", fontsize=15); ax.set_ylabel("test error (lower is better)", fontsize=15)
ax.legend(fontsize=13, loc="upper left", frameon=False); fig.tight_layout(); fig.savefig("fig_poster_solver.png")

# architecture sweeps, 2 x 2
fig, axs = plt.subplots(2, 2, figsize=(COLW, 6.6), sharey=True)
F.sweep("e2_modes", "modes", "Fourier modes per layer", axs[0, 0], int); F.sweep("e2_width", "w", "width (channels)", axs[0, 1], int)
F.sweep("e2_layers", "l", "layers", axs[1, 0], int); F.sweep("e2_steps", "s", "training steps", axs[1, 1], int)
axs[1, 1].set_xscale("log"); axs[1, 1].set_xticks([5700, 20000, 80000]); axs[1, 1].set_xticklabels(["5.7k", "20k", "80k"]); axs[1, 1].minorticks_off()
axs[0, 0].set_ylabel("rel. L2 error"); axs[1, 0].set_ylabel("rel. L2 error"); axs[0, 0].legend(fontsize=13)
floor_fonts(fig, 13); fig.tight_layout(); fig.savefig("fig_poster_arch.png")

# data scaling + resolution
fig, axs = plt.subplots(1, 2, figsize=(COLW, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
for model, c in (("fno", GREEN), ("mlp", ORANGE)):
    d = defaultdict(list)
    for n, r in F.rows.items():
        m = re.match(rf"e3_{model}_n(\d+)_s\d", n)
        if m: d[int(m.group(1))].append(r["tta"])
    ns = sorted(d); ax.errorbar(ns, [np.mean(d[n]) for n in ns], [np.std(d[n]) for n in ns], fmt="o-", color=c, label=model.upper(), capsize=3, lw=2, ms=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training maps"); ax.set_ylabel("test error"); ax.legend(fontsize=13)
ax.axvline(900, color=GREY, ls=":"); ax.set_title("more data", fontsize=15, color=NAVY)
M = np.full((3, 3), np.nan); rs = [32, 64, 128]
for i, tr in enumerate(rs):
    for j, te in enumerate(rs):
        v = [r["test"] if te == tr else (r["at_res"] or {}).get(str(te)) for n, r in F.rows.items() if re.fullmatch(rf"e4_fno_res{tr}_s\d", n)]
        M[i, j] = np.mean(v)
ax = axs[1]; ax.imshow(M, cmap="YlOrRd"); ax.grid(False)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=13, fontweight="bold" if i == j else None)
ax.set_xticks(range(3)); ax.set_xticklabels(rs); ax.set_yticks(range(3)); ax.set_yticklabels(rs)
ax.set_xlabel("test grid"); ax.set_ylabel("training grid"); ax.set_title("other grids", fontsize=15, color=NAVY)
fig.tight_layout(); fig.savefig("fig_poster_data.png")

# example predictions (best / median / worst) and the single best row
d = np.load(ROOT / "results" / "stage3" / "diagnostics_arrays.npz"); rel = d["rel"]
idx = [int(np.argmin(rel)), int(np.argsort(rel)[len(rel) // 2]), int(np.argmax(rel))]
def panels(axrow, i, show_err=True):
    vm = d["gt"][i].max()
    for c, (a, t, cm, kw) in enumerate(((d["kappa"][i], "permeability", "gray", {}), (d["gt"][i], "truth", "viridis", dict(vmin=0, vmax=vm)),
                                        (d["pred"][i], "FNO", "viridis", dict(vmin=0, vmax=vm)), (np.abs(d["pred"][i] - d["gt"][i]), "|error|", "magma", {}))):
        im = axrow[c].imshow(a, cmap=cm, origin="lower", **kw); axrow[c].set_xticks([]); axrow[c].set_yticks([]); axrow[c].grid(False)
        axrow[c].set_title(t + (f"  ({rel[i]*100:.1f}%)" if c == 0 and show_err else ""), fontsize=13)
        cb = plt.colorbar(im, ax=axrow[c], fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=10)
fig, axs = plt.subplots(3, 4, figsize=(COLW * 0.92, 7.0))
for r_, i in enumerate(idx): panels(axs[r_], i)
fig.tight_layout(); fig.savefig("fig_poster_samples.png")
fig, axs = plt.subplots(1, 4, figsize=(COLW, 2.6))
panels(axs, idx[1], show_err=False); fig.tight_layout(); fig.savefig("fig_poster_samplerow.png")
print("poster redraws done")

# recipe ablation, poster layout (same e1 runs as the report figure; bars = mean test error of 3 seeds, whiskers = sd)
g = defaultdict(list)
for n, r in F.rows.items():
    m = re.match(r"e1_(\w+?)_(leg|cos)_(mse|rel_l2)_(none|d4)_s\d", n)
    if m: g[m.groups()].append(r)
order = [("leg", "mse", "none", "Stage 2\nrecipe"), ("cos", "mse", "none", "+ single\ncosine"), ("leg", "rel_l2", "none", "+ rel-L2\nloss only"),
         ("leg", "mse", "d4", "+ D4 aug\nonly"), ("cos", "rel_l2", "none", "cosine\n+ rel-L2"), ("cos", "rel_l2", "d4", "all three")]
fig, ax = plt.subplots(figsize=(COLW * 0.96, 5.0))
x = np.arange(len(order)); w = 0.4
for j, (model, c) in enumerate((("fno", GREEN), ("mlp", ORANGE))):
    mm = [np.mean([r["test"] for r in g[(model,) + o[:3]]]) for o in order]
    sd = [np.std([r["test"] for r in g[(model,) + o[:3]]], ddof=1) for o in order]
    ax.bar(x + (j - .5) * w, mm, w, yerr=sd, color=c, label=model.upper(), capsize=3)
    for xi, v, e in zip(x + (j - .5) * w, mm, sd):
        ax.text(xi, v + e + 0.0015, f"{v:.3f}", ha="center", fontsize=11.5, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([o[3] for o in order], fontsize=13); ax.set_ylim(0, 0.105)
ax.set_ylabel("test error (3 seeds)", fontsize=14); ax.legend(fontsize=13, loc="upper left", ncol=2, frameon=False)
fig.tight_layout(); fig.savefig("fig_poster_recipe_ablation.png")
print("recipe redraw done")
