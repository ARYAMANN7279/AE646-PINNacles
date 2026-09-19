"""
Exploratory data analysis of the real PDEBench Darcy split used in this project
(data/processed/*.npz, seed 42 - see src/preprocess.py).

Reports: the permeability value distribution, the high-permeability area fraction per
sample, per-sample peak pressure, and the number of connected high-permeability regions
in each test sample at native 128x128 resolution.

Run: python src/eda.py
Requires: data/processed/{train,val,test,test_hires}.npz and norm_stats.json
Writes:   results/eda_metrics.json, results/figures/fig8_eda_dataset.{png,pdf}
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def load_physical(data_dir, split, tensor_mean, tensor_std, coeff_mean, coeff_std):
    """Load a preprocessed split and undo the standardisation (physical units)."""
    d = np.load(Path(data_dir) / f"{split}.npz")
    coeff = d["inputs"][..., 0] * coeff_std + coeff_mean
    tensor = d["targets"][..., 0] * tensor_std + tensor_mean
    return coeff, tensor


def count_regions(coeff_hires):
    """Number of connected high-permeability regions in each field."""
    counts = []
    for k in coeff_hires:
        _, n = ndimage.label(k > k.mean())
        counts.append(n)
    return np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-json", default="results/eda_metrics.json")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "norm_stats.json") as f:
        stats = json.load(f)
    cm, cs = stats["coeff_mean"], stats["coeff_std"]
    tm, ts = stats["tensor_mean"], stats["tensor_std"]

    splits = [load_physical(data_dir, s, tm, ts, cm, cs) for s in ("train", "val", "test")]
    all_coeff = np.concatenate([c for c, _ in splits])
    all_tensor = np.concatenate([t for _, t in splits])

    hires = np.load(data_dir / "test_hires.npz")
    n_regions = count_regions(hires["coeff"])

    # kappa is bimodal at {0.1, 1.0}, so 0.55 cleanly separates high from low permeability
    frac_high = (all_coeff > 0.55).mean(axis=(1, 2))
    peak = all_tensor.max(axis=(1, 2))
    summary = {
        "n_samples_total": int(len(all_coeff)),
        "n_train": int(len(splits[0][0])), "n_val": int(len(splits[1][0])), "n_test": int(len(splits[2][0])),
        "kappa_unique_values_approx": sorted({round(float(v), 3) for v in np.unique(all_coeff)[:5]}),
        "kappa_mean": float(all_coeff.mean()), "kappa_std": float(all_coeff.std()),
        "high_kappa_fraction_mean": float(frac_high.mean()), "high_kappa_fraction_std": float(frac_high.std()),
        "pressure_mean": float(all_tensor.mean()), "pressure_std": float(all_tensor.std()),
        "pressure_max_mean": float(peak.mean()), "pressure_max_std": float(peak.std()),
        "test_region_count_histogram_native_128": {str(k): int(v) for k, v in sorted(Counter(n_regions.tolist()).items())},
    }
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].hist(all_coeff.ravel(), bins=50, color="#4C72B0")
    axes[0, 0].set_title("$\\kappa$ value distribution\n(all 1200 samples, physical units)", fontsize=11)
    axes[0, 0].set_xlabel("$\\kappa$"); axes[0, 0].set_ylabel("Grid-point count"); axes[0, 0].set_yscale("log")

    axes[0, 1].hist(frac_high, bins=30, color="#55A868")
    axes[0, 1].set_title("High-permeability area fraction\nper sample ($\\kappa$=1.0 coverage)", fontsize=11)
    axes[0, 1].set_xlabel("High-$\\kappa$ area fraction"); axes[0, 1].set_ylabel("Sample count")

    axes[1, 0].hist(peak, bins=30, color="#C44E52")
    axes[1, 0].set_title("Per-sample peak pressure $\\max(u)$", fontsize=11)
    axes[1, 0].set_xlabel("Peak pressure (physical units)"); axes[1, 0].set_ylabel("Sample count")

    bins = np.arange(n_regions.min(), n_regions.max() + 2) - 0.5
    axes[1, 1].hist(n_regions, bins=bins, color="#8172B2", rwidth=0.7)
    axes[1, 1].set_xticks(sorted(set(n_regions.tolist())))
    axes[1, 1].set_title("Connected high-$\\kappa$ regions\nper test sample (native $128^2$)", fontsize=11)
    axes[1, 1].set_xlabel("# connected regions"); axes[1, 1].set_ylabel("Sample count")
    fig.tight_layout()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out_dir}/fig8_eda_dataset.png", dpi=150)
    fig.savefig(f"{args.out_dir}/fig8_eda_dataset.pdf")
    plt.close(fig)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"dataset_summary": summary}, f, indent=2)
    print(f"Saved -> {args.out_json}, {args.out_dir}/fig8_eda_dataset.png")


if __name__ == "__main__":
    main()
