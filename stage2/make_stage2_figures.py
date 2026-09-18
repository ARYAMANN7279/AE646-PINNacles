"""
Builds the Stage-2-only figures in stage2/figures/ from the stored results
(MLP baseline + original FNO only), so the interim report and deck show
exactly the Stage 2 scope.

Run (from repo root):  python3 stage2/make_stage2_figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

C_FNO, C_MLP = "#2E86C1", "#CA6F1E"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})


def errors(run):
    with open(RES / run / "evaluation" / "eval_metrics.json") as f:
        return np.array(json.load(f)["sample_rel_l2"])


# Fig 1: error histogram, MLP baseline vs original FNO only
fno, mlp = errors("run_001"), errors("run_002")
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 0.35, 35)
ax.hist(fno, bins=bins, alpha=0.65, color=C_FNO, edgecolor="white", linewidth=0.4,
        label=f"FNO (mean={fno.mean():.3f})")
ax.hist(mlp, bins=bins, alpha=0.65, color=C_MLP, edgecolor="white", linewidth=0.4,
        label=f"MLP baseline (mean={mlp.mean():.3f})")
for m, c in [(fno.mean(), C_FNO), (mlp.mean(), C_MLP)]:
    ax.axvline(m, color=c, linestyle="--", linewidth=1.5, alpha=0.9)
ax.set_xlabel("Relative L2 Error", fontsize=12)
ax.set_ylabel("Number of Test Samples", fontsize=12)
ax.set_title("Error Distribution - 200 Test Samples (64x64)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig1_error_histogram.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_error_histogram.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# Fig: FNO predictions for the best, a median-error and the worst test sample
# (top 3 rows of the evaluation figure: samples 17 / 191 / 95)
img = mpimg.imread(RES / "run_001" / "evaluation" / "sample_predictions.png")
h = img.shape[0]
crop = img[: int(round(h * 3 / 8))]
fig, ax = plt.subplots(figsize=(8, 8 * crop.shape[0] / crop.shape[1]))
ax.imshow(crop)
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT / "fig_sample_predictions.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# Figures produced by the EDA / ablation scripts, copied so this folder is self-contained
for name in ["fig8_eda_dataset", "fig7_mlp_ablation", "fig10_preprocess_ablation"]:
    for ext in ("pdf", "png"):
        src = RES / "figures" / f"{name}.{ext}"
        if src.exists():
            shutil.copy(src, OUT / src.name)
print("Stage 2 figures written to", OUT)
