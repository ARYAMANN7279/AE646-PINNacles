"""
Generate all report-quality figures for PINNacles AE646 project.
Outputs go to results/figures/

Run: python src/generate_figures.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES    = os.path.join(ROOT, "results")
OUTDIR = os.path.join(RES, "figures")
os.makedirs(OUTDIR, exist_ok=True)

def out(name): return os.path.join(OUTDIR, name)

# ── load data ──────────────────────────────────────────────────────────────────
def load_eval(run):
    with open(os.path.join(RES, run, "evaluation", "eval_metrics.json")) as f:
        return json.load(f)

def load_test(run):
    with open(os.path.join(RES, run, "test_metrics.json")) as f:
        return json.load(f)

fno_eval  = load_eval("run_001")
mlp_eval  = load_eval("run_002")
fnoi_eval = load_eval("run_003_fno_improved")

fno_test  = load_test("run_001")
mlp_test  = load_test("run_002")
fnoi_test = load_test("run_003_fno_improved")

with open(os.path.join(RES, "run_001", "superres_metrics.json")) as f:
    fno_sr = json.load(f)
with open(os.path.join(RES, "run_003_fno_improved", "superres_metrics.json")) as f:
    fnoi_sr = json.load(f)
with open(os.path.join(RES, "benchmark_speed.json")) as f:
    speed = json.load(f)
with open(os.path.join(RES, "benchmark_components.json")) as f:
    comp = json.load(f)

fno_errors  = np.array(fno_eval["sample_rel_l2"])
mlp_errors  = np.array(mlp_eval["sample_rel_l2"])
fnoi_errors = np.array(fnoi_eval["sample_rel_l2"])

# colour palette
C_FNO  = "#2E86C1"   # blue
C_FNOI = "#1E8B4C"   # green
C_MLP  = "#CA6F1E"   # orange
C_GRAY = "#7F8C8D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

print("Generating figures...")

# ── Fig 1: Combined error histogram ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 0.35, 35)
ax.hist(fno_errors,  bins=bins, alpha=0.65, color=C_FNO,  label=f"FNO original  (mean={fno_errors.mean():.3f})", edgecolor="white", linewidth=0.4)
ax.hist(fnoi_errors, bins=bins, alpha=0.65, color=C_FNOI, label=f"FNO improved  (mean={fnoi_errors.mean():.3f})", edgecolor="white", linewidth=0.4)
ax.hist(mlp_errors,  bins=bins, alpha=0.65, color=C_MLP,  label=f"MLP baseline  (mean={mlp_errors.mean():.3f})", edgecolor="white", linewidth=0.4)
for mean, c in [(fno_errors.mean(), C_FNO), (fnoi_errors.mean(), C_FNOI), (mlp_errors.mean(), C_MLP)]:
    ax.axvline(mean, color=c, linestyle="--", linewidth=1.5, alpha=0.9)
ax.set_xlabel("Relative L2 Error", fontsize=12)
ax.set_ylabel("Number of Test Samples", fontsize=12)
ax.set_title("Error Distribution — 200 Test Samples (64×64)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(out("fig1_error_histogram.pdf"), bbox_inches="tight")
fig.savefig(out("fig1_error_histogram.png"), bbox_inches="tight")
plt.close()
print("  fig1_error_histogram done")

# ── Fig 2: Efficiency scatter (params vs Rel L2) ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
params = {"MLP": 42.0, "FNO original": 4.7, "FNO improved": 78.8}
means  = {
    "MLP":          mlp_errors.mean(),
    "FNO original": fno_errors.mean(),
    "FNO improved": fnoi_errors.mean(),
}
colors = {"MLP": C_MLP, "FNO original": C_FNO, "FNO improved": C_FNOI}
for label in params:
    ax.scatter(params[label], means[label], s=160, color=colors[label],
               zorder=5, edgecolors="white", linewidths=1.2)
    offset = {"MLP": (2, 0.002), "FNO original": (-1.5, 0.003), "FNO improved": (2, 0.001)}
    ax.annotate(label, xy=(params[label], means[label]),
                xytext=(params[label]+offset[label][0], means[label]+offset[label][1]),
                fontsize=10, color=colors[label], fontweight="bold")
ax.set_xlabel("Parameter Count (millions)", fontsize=12)
ax.set_ylabel("Mean Relative L2 Error (64×64 test)", fontsize=12)
ax.set_title("Efficiency: Accuracy vs. Model Size", fontsize=13, fontweight="bold")
ax.set_xlim(-5, 90)
ax.set_ylim(0.03, 0.10)
plt.tight_layout()
fig.savefig(out("fig2_efficiency_scatter.pdf"), bbox_inches="tight")
fig.savefig(out("fig2_efficiency_scatter.png"), bbox_inches="tight")
plt.close()
print("  fig2_efficiency_scatter done")

# ── Fig 3: Zero-shot super-resolution comparison ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
models = ["FNO original\n64×64 (train)", "FNO original\n128×128 (zero-shot)",
          "FNO improved\n64×64 (train)", "FNO improved\n128×128 (zero-shot)"]
vals   = [fno_errors.mean(), fno_sr["mean_rel_l2"],
          fnoi_errors.mean(), fnoi_sr["mean_rel_l2"]]
errs   = [fno_errors.std(), fno_sr["std_rel_l2"],
          fnoi_errors.std(), fnoi_sr["std_rel_l2"]]
clrs   = [C_FNO, C_FNO, C_FNOI, C_FNOI]
alphas = [0.9, 0.55, 0.9, 0.55]
bars = ax.bar(models, vals, color=clrs, alpha=0.85,
              edgecolor="white", linewidth=0.8, width=0.5)
ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="none",
            color="black", capsize=4, linewidth=1.2, alpha=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Mean Relative L2 Error", fontsize=11)
ax.set_title("Zero-Shot Super-Resolution (MLP cannot do this)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.10)
ax.axhline(mlp_errors.mean(), color=C_MLP, linestyle="--", linewidth=1.5,
           label=f"MLP baseline (64×64) = {mlp_errors.mean():.3f}")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(out("fig3_superres.pdf"), bbox_inches="tight")
fig.savefig(out("fig3_superres.png"), bbox_inches="tight")
plt.close()
print("  fig3_superres done")

# ── Fig 4: Inference speed comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
spd_labels = ["MLP\n(42M params)", "FNO original\n(4.7M params)", "FDM solver\n(reference)"]
spd_vals   = [speed["mlp_ms_per_sample"], speed["fno_ms_per_sample"], speed["fdm_ms_per_sample"]]
spd_colors = [C_MLP, C_FNO, C_GRAY]
bars = ax.bar(spd_labels, spd_vals, color=spd_colors, alpha=0.85,
              edgecolor="white", linewidth=0.8, width=0.45)
for bar, v in zip(bars, spd_vals):
    label = f"{v:.2f} ms" if v < 10 else f"{v:.0f} ms"
    ax.text(bar.get_x() + bar.get_width()/2, v + 15, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Inference Time (ms per sample)", fontsize=11)
ax.set_title("Inference Speed vs. Traditional Solver", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.05, 5000)
plt.tight_layout()
fig.savefig(out("fig4_speed.pdf"), bbox_inches="tight")
fig.savefig(out("fig4_speed.png"), bbox_inches="tight")
plt.close()
print("  fig4_speed done")

# ── Fig 5: Component timing breakdown (bar) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart
pct = comp["pct"]
comp_labels  = ["Lift\n(fc0)", "FFT\n(×4)", "Cmplx\nMul (×4)", "IFFT\n(×4)", "Local\nConv×4", "Project\n(fc1+fc2)"]
comp_pcts    = [pct["lift"], pct["fft"], pct["complex_mul"], pct["ifft"], pct["local_conv"], pct["projection"]]
comp_colors  = [C_GRAY, C_FNO, "#1A2E4A", C_FNO, C_FNOI, C_GRAY]
axes[0].bar(comp_labels, comp_pcts, color=comp_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
for i, v in enumerate(comp_pcts):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_ylabel("% of Total Component Time", fontsize=10)
axes[0].set_title("FNO Component Time Breakdown", fontsize=11, fontweight="bold")

# Pie chart
pie_labels = ["FFT", "Complex Mul", "IFFT", "Local Conv", "Lift+Project"]
pie_vals   = [pct["fft"], pct["complex_mul"], pct["ifft"], pct["local_conv"],
              pct["lift"] + pct["projection"]]
pie_colors = [C_FNO, "#1A2E4A", "#5DADE2", C_FNOI, C_GRAY]
wedges, texts, autotexts = axes[1].pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    textprops={"fontsize": 9}
)
for at in autotexts: at.set_fontsize(8)
axes[1].set_title("Spectral ops = 68.4% of compute", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(out("fig5_components.pdf"), bbox_inches="tight")
fig.savefig(out("fig5_components.png"), bbox_inches="tight")
plt.close()
print("  fig5_components done")

# ── Fig 6: Summary comparison table as figure ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")
rows = [
    ["MLP baseline",    "42.0M", f"{mlp_errors.mean():.4f}",  f"{np.median(mlp_errors):.4f}",
     f"{mlp_errors.std():.4f}", "N/A", f"{speed['mlp_ms_per_sample']:.2f}"],
    ["FNO original",    "4.7M",  f"{fno_errors.mean():.4f}",  f"{np.median(fno_errors):.4f}",
     f"{fno_errors.std():.4f}", f"{fno_sr['mean_rel_l2']:.4f}", f"{speed['fno_ms_per_sample']:.2f}"],
    ["FNO improved",    "78.8M", f"{fnoi_errors.mean():.4f}", f"{np.median(fnoi_errors):.4f}",
     f"{fnoi_errors.std():.4f}", f"{fnoi_sr['mean_rel_l2']:.4f}", "~0.71"],
]
cols = ["Model", "Params", "Mean Rel L2\n(64×64)", "Median Rel L2\n(64×64)",
        "Std Dev", "Mean Rel L2\n(128×128, zero-shot)", "Speed\n(ms/sample)"]
tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1A2E4A")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    elif r == 2:   # FNO original — highlight
        cell.set_facecolor("#D6E4F0")
    elif r == 3:   # FNO improved
        cell.set_facecolor("#D5F5E3")
    cell.set_edgecolor("#BDC3C7")
ax.set_title("Model Comparison — Quantitative Results", fontsize=13,
             fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(out("fig6_comparison_table.pdf"), bbox_inches="tight")
fig.savefig(out("fig6_comparison_table.png"), bbox_inches="tight")
plt.close()
print("  fig6_comparison_table done")

print(f"\nAll figures saved to: {OUTDIR}/")
