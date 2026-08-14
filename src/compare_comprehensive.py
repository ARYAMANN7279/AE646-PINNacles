"""
Comprehensive comparison of all models for final report.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(run_dir):
    with open(run_dir / "eval_metrics.json", "r") as f:
        return json.load(f)


def create_comprehensive_comparison():
    """Create comprehensive comparison with all three models."""
    fno_metrics = load_metrics(Path("results/run_001/evaluation"))
    mlp_metrics = load_metrics(Path("results/run_002/evaluation"))
    fno_improved_metrics = load_metrics(Path("results/run_003_fno_improved/evaluation"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = ["FNO (Original)", "MLP Baseline", "FNO (Improved)"]
    colors = ["blue", "orange", "green"]
    metrics_list = [fno_metrics, mlp_metrics, fno_improved_metrics]

    # 1. Error distribution comparison (all three)
    ax = axes[0, 0]
    for i, (m, c, label) in enumerate(zip(metrics_list, colors, models)):
        ax.hist(m["sample_rel_l2"], bins=20, alpha=0.5, label=label, density=True, 
                edgecolor=c, color=c)
        ax.axvline(m["mean_rel_l2"], color=c, linestyle="--", 
                   label=f"{label} mean: {m['mean_rel_l2']:.4f}")
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution Comparison (All Models)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # 2. Bar chart of key metrics
    ax = axes[0, 1]
    mean_rel_l2 = [fno_metrics["mean_rel_l2"], mlp_metrics["mean_rel_l2"], fno_improved_metrics["mean_rel_l2"]]
    median_rel_l2 = [fno_metrics["median_rel_l2"], mlp_metrics["median_rel_l2"], fno_improved_metrics["median_rel_l2"]]
    std_rel_l2 = [fno_metrics["std_rel_l2"], mlp_metrics["std_rel_l2"], fno_improved_metrics["std_rel_l2"]]

    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, mean_rel_l2, width, label="Mean", color=colors, edgecolor="black")
    ax.bar(x, median_rel_l2, width, label="Median", color=colors, edgecolor="black", alpha=0.7)
    ax.bar(x + width, std_rel_l2, width, label="Std", color=colors, edgecolor="black", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.set_yscale("log")

    # 3. Min/Max comparison
    ax = axes[0, 2]
    min_vals = [fno_metrics["min_rel_l2"], mlp_metrics["min_rel_l2"], fno_improved_metrics["min_rel_l2"]]
    max_vals = [fno_metrics["max_rel_l2"], mlp_metrics["max_rel_l2"], fno_improved_metrics["max_rel_l2"]]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, min_vals, width, label="Min", color="green", edgecolor="black", alpha=0.7)
    ax.bar(x + width/2, max_vals, width, label="Max", color="red", edgecolor="black", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Min/Max Error Comparison")
    ax.legend()
    ax.set_yscale("log")

    # 4. Model parameters comparison
    ax = axes[1, 0]
    params = [4_744_449, 41_953_280, 78_760_961]
    bars = ax.bar(models, params, color=colors, edgecolor="black")
    ax.set_ylabel("Parameters")
    ax.set_title("Model Size Comparison")
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e6,
                f"{p/1e6:.1f}M", ha="center", va="bottom", fontweight="bold")
    ax.set_yscale("log")

    # 5. Error vs Parameters scatter (efficiency)
    ax = axes[1, 1]
    for i, (m, c, label) in enumerate(zip(metrics_list, colors, models)):
        ax.scatter(params[i], m["mean_rel_l2"], color=c, s=200, label=label, edgecolor="black", zorder=5)
        ax.annotate(f"{m['mean_rel_l2']:.4f}", (params[i], m["mean_rel_l2"]), 
                    textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Mean Relative L2 Error")
    ax.set_title("Efficiency: Error vs Model Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cumulative error distribution
    ax = axes[1, 2]
    for m, c, label in zip(metrics_list, colors, models):
        sorted_errors = np.sort(m["sample_rel_l2"])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, color=c, label=label, linewidth=2)
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Error Distribution")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/comprehensive_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Comprehensive comparison saved to results/comprehensive_comparison.png")


def print_comprehensive_summary():
    fno = load_metrics(Path("results/run_001/evaluation"))
    mlp = load_metrics(Path("results/run_002/evaluation"))
    fno_imp = load_metrics(Path("results/run_003_fno_improved/evaluation"))

    print("=" * 80)
    print("COMPREHENSIVE FINAL PROJECT SUMMARY: Darcy Flow Surrogate Modeling")
    print("=" * 80)
    print()
    print(f"{'Metric':<30} {'FNO (Original)':>14} {'MLP Baseline':>14} {'FNO (Improved)':>14}")
    print("-" * 80)
    print(f"{'Mean Rel L2':<30} {fno['mean_rel_l2']:>14.4f} {mlp['mean_rel_l2']:>14.4f} {fno_imp['mean_rel_l2']:>14.4f}")
    print(f"{'Median Rel L2':<30} {fno['median_rel_l2']:>14.4f} {mlp['median_rel_l2']:>14.4f} {fno_imp['median_rel_l2']:>14.4f}")
    print(f"{'Std Rel L2':<30} {fno['std_rel_l2']:>14.4f} {mlp['std_rel_l2']:>14.4f} {fno_imp['std_rel_l2']:>14.4f}")
    print(f"{'Min Rel L2':<30} {fno['min_rel_l2']:>14.4f} {mlp['min_rel_l2']:>14.4f} {fno_imp['min_rel_l2']:>14.4f}")
    print(f"{'Max Rel L2':<30} {fno['max_rel_l2']:>14.4f} {mlp['max_rel_l2']:>14.4f} {fno_imp['max_rel_l2']:>14.4f}")
    print(f"{'Parameters':<30} {'4.7M':>14} {'42.0M':>14} {'78.8M':>14}")
    print(f"{'Error/Param (x1e-6)':<30} {fno['mean_rel_l2']/4.7:>14.2f} {mlp['mean_rel_l2']/42.0:>14.2f} {fno_imp['mean_rel_l2']/78.8:>14.2f}")
    print()
    print("Key Findings:")
    print(f"  - Original FNO achieves {mlp['mean_rel_l2']/fno['mean_rel_l2']:.0f}x lower error than MLP with 8.9x fewer parameters")
    print(f"  - Improved FNO achieves {mlp['mean_rel_l2']/fno_imp['mean_rel_l2']:.0f}x lower error than MLP but with 1.9x MORE parameters")
    print(f"  - Original FNO is most efficient: {fno['mean_rel_l2']/4.7:.2e} error per million params")
    print(f"  - FNO error distribution is tight (std={fno['std_rel_l2']:.4f}) vs MLP (std={mlp['std_rel_l2']:.4f})")
    print(f"  - Literature reports ~0.01-0.02 rel L2 for FNO on Darcy (this work: {fno['mean_rel_l2']:.3f})")
    print("=" * 80)


if __name__ == "__main__":
    create_comprehensive_comparison()
    print_comprehensive_summary()