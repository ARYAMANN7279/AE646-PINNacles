import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# -------------------------------------------------------------
# Subplot 1 (Top-Left): Permeability Distribution (FIXED & HONEST)
# -------------------------------------------------------------
# Discrete bimodal values: kappa in {0.1, 1.0}
x_pos = np.array([0.1, 1.0])
counts = np.array([2553425, 2361775]) # 51.95% vs 48.05%
percentages = counts / counts.sum() * 100

bars = axes[0, 0].bar(x_pos, counts, width=0.22, color="#4C72B0", edgecolor="#2B4C7E", linewidth=1.2, zorder=3)
axes[0, 0].set_title(r"$\kappa$ value distribution" + "\n(all 1200 samples, physical units)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel(r"Permeability $\kappa$", fontsize=10)
axes[0, 0].set_ylabel("Grid-point count", fontsize=10)
axes[0, 0].set_xticks([0.1, 1.0])
axes[0, 0].set_xticklabels([r"$\kappa = 0.1$" + "\n(dense)", r"$\kappa = 1.0$" + "\n(porous)"], fontsize=9.5)
axes[0, 0].set_xlim(-0.15, 1.25)
axes[0, 0].set_ylim(0, 3.2e6)
axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)

# Format y-axis in Millions
from matplotlib.ticker import FuncFormatter
axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*1e-6:.1f}M"))

# Add exact count and percentage labels above each bar
for bar, count, pct in zip(bars, counts, percentages):
    yval = bar.get_height()
    axes[0, 0].text(
        bar.get_x() + bar.get_width()/2.0, yval + 70000,
        f"{count*1e-6:.2f}M\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1B365D"
    )

# -------------------------------------------------------------
# Subplot 2 (Top-Right): High-Permeability Area Fraction
# -------------------------------------------------------------
green_counts = [26, 21, 19, 17, 21, 24, 28, 37, 32, 28, 53, 63, 60, 61, 64, 75, 75, 65, 79, 77, 61, 49, 46, 35, 28, 22, 12, 11, 8, 3]
bin_edges_green = np.linspace(0.0000, 0.9607, 31)
axes[0, 1].hist(bin_edges_green[:-1], bins=bin_edges_green, weights=green_counts, color="#55A868", edgecolor="#3A7547", linewidth=0.5)
axes[0, 1].set_title(r"High-permeability area fraction" + "\n" + r"per sample ($\kappa$=1.0 coverage)", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel(r"High-$\kappa$ area fraction", fontsize=10)
axes[0, 1].set_ylabel("Sample count", fontsize=10)
axes[0, 1].set_xlim(-0.02, 1.02)
axes[0, 1].set_ylim(0, 85)

# -------------------------------------------------------------
# Subplot 3 (Bottom-Left): Per-Sample Peak Pressure
# -------------------------------------------------------------
red_counts = [29, 65, 107, 165, 185, 140, 113, 83, 61, 45, 26, 28, 22, 13, 15, 18, 14, 12, 12, 12, 3, 10, 3, 3, 6, 4, 1, 0, 3, 2]
bin_edges_red = np.linspace(0.0770, 1.2352, 31)
axes[1, 0].hist(bin_edges_red[:-1], bins=bin_edges_red, weights=red_counts, color="#C44E52", edgecolor="#8B3336", linewidth=0.5)
axes[1, 0].set_title(r"Per-sample peak pressure $\max(u)$", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Peak pressure (physical units)", fontsize=10)
axes[1, 0].set_ylabel("Sample count", fontsize=10)
axes[1, 0].set_xlim(0.05, 1.3)
axes[1, 0].set_ylim(0, 195)

# -------------------------------------------------------------
# Subplot 4 (Bottom-Right): Connected High-kappa Regions
# -------------------------------------------------------------
n_regions = [0, 1, 2]
purple_counts = [2, 180, 18]
axes[1, 1].bar(n_regions, purple_counts, color="#8172B2", edgecolor="#594B82", width=0.7)
axes[1, 1].set_xticks(n_regions)
axes[1, 1].set_title(r"Connected high-$\kappa$ regions" + "\n" + r"per test sample (native $128^2$)", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel(r"# connected regions", fontsize=10)
axes[1, 1].set_ylabel("Sample count", fontsize=10)
axes[1, 1].set_ylim(0, 195)

fig.tight_layout()

# Save to all required destinations
out_paths = [
    Path("stage2/figures/fig8_eda_dataset.png"),
    Path("stage2/figures/fig8_eda_dataset.pdf"),
    Path("results/figures/fig8_eda_dataset.png"),
    Path("results/figures/fig8_eda_dataset.pdf"),
    Path("stage3/figures/fig8_eda_dataset.pdf"),
    Path("stage2/PINNacles_Stage2_Code/results/figures/fig8_eda_dataset.png"),
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".png":
        fig.savefig(p, dpi=200)
    elif p.suffix == ".pdf":
        fig.savefig(p)
    print(f"Saved -> {p}")

plt.close(fig)
print("Finished updating all EDA figures!")
