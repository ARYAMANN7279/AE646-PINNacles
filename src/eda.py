"""
Exploratory data analysis on the real PDEBench Darcy split actually used by
this project (data/processed/*.npz, seed=42 - see src/preprocess.py), plus a
quantitative version of the "high-error samples have complex kappa fields"
claim made qualitatively in the reports: correlates each test sample's kappa
heterogeneity (number of connected permeability regions, interface length)
against the trained FNO's per-sample test error.

Run: python src/eda.py
Requires: data/processed/{train,val,test,test_hires}.npz, norm_stats.json,
          results/run_001/evaluation/eval_metrics.json (FNO original errors)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def load_physical(data_dir, split, tensor_mean, tensor_std, coeff_mean, coeff_std):
    d = np.load(Path(data_dir) / f"{split}.npz")
    coeff_n = d["inputs"][..., 0]
    tensor_n = d["targets"][..., 0]
    coeff = coeff_n * coeff_std + coeff_mean
    tensor = tensor_n * tensor_std + tensor_mean
    return coeff, tensor


def heterogeneity_metrics(coeff_hires):
    """Per-sample: number of connected high-permeability regions, and
    interface perimeter (count of adjacent pixel pairs that straddle a
    permeability jump), on the native-resolution kappa field."""
    n_regions, perimeter = [], []
    for k in coeff_hires:
        binary = (k > k.mean())
        _, n = ndimage.label(binary)
        n_regions.append(n)
        vert = np.sum(binary[1:, :] != binary[:-1, :])
        horiz = np.sum(binary[:, 1:] != binary[:, :-1])
        perimeter.append(int(vert + horiz))
    return np.array(n_regions), np.array(perimeter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--fno-eval", default="results/run_001/evaluation/eval_metrics.json")
    ap.add_argument("--out-json", default="results/eda_metrics.json")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "norm_stats.json") as f:
        stats = json.load(f)
    cm, cs, tm, ts = stats["coeff_mean"], stats["coeff_std"], stats["tensor_mean"], stats["tensor_std"]

    train_coeff, train_tensor = load_physical(data_dir, "train", tm, ts, cm, cs)
    val_coeff, val_tensor = load_physical(data_dir, "val", tm, ts, cm, cs)
    test_coeff, test_tensor = load_physical(data_dir, "test", tm, ts, cm, cs)
    all_coeff = np.concatenate([train_coeff, val_coeff, test_coeff])
    all_tensor = np.concatenate([train_tensor, val_tensor, test_tensor])
    n_total = len(all_coeff)

    hires = np.load(data_dir / "test_hires.npz")
    coeff_hires = hires["coeff"]  # (200, 128, 128), real physical, native resolution

    # --- dataset-level summary ---
    # kappa is bimodal at {0.1, 1.0} (verified below), so thresholding at the
    # midpoint (0.55) cleanly separates high- from low-permeability grid points.
    frac_high = (all_coeff > 0.55).mean(axis=(1, 2))
    summary = {
        "n_samples_total": int(n_total),
        "n_train": int(len(train_coeff)), "n_val": int(len(val_coeff)), "n_test": int(len(test_coeff)),
        "kappa_unique_values_approx": sorted({round(float(v), 3) for v in np.unique(all_coeff)[:5]}),
        "kappa_mean": float(all_coeff.mean()), "kappa_std": float(all_coeff.std()),
        "high_kappa_fraction_mean": float(frac_high.mean()), "high_kappa_fraction_std": float(frac_high.std()),
        "pressure_mean": float(all_tensor.mean()), "pressure_std": float(all_tensor.std()),
        "pressure_max_mean": float(all_tensor.max(axis=(1, 2)).mean()),
        "pressure_max_std": float(all_tensor.max(axis=(1, 2)).std()),
    }
    print("Dataset summary:", json.dumps(summary, indent=2))

    # --- figure: dataset EDA (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].hist(all_coeff.ravel(), bins=50, color="#4C72B0")
    axes[0, 0].set_title("$\\kappa$ value distribution\n(all 1200 samples, physical units)", fontsize=11)
    axes[0, 0].set_xlabel("$\\kappa$"); axes[0, 0].set_ylabel("Grid-point count")
    axes[0, 0].set_yscale("log")

    axes[0, 1].hist(frac_high, bins=30, color="#55A868")
    axes[0, 1].set_title("High-permeability area fraction\nper sample ($\\kappa$=1.0 coverage)", fontsize=11)
    axes[0, 1].set_xlabel("High-$\\kappa$ area fraction"); axes[0, 1].set_ylabel("Sample count")

    axes[1, 0].hist(all_tensor.max(axis=(1, 2)), bins=30, color="#C44E52")
    axes[1, 0].set_title("Per-sample peak pressure $\\max(u)$", fontsize=11)
    axes[1, 0].set_xlabel("Peak pressure (physical units)"); axes[1, 0].set_ylabel("Sample count")

    n_regions_all, perimeter_all = heterogeneity_metrics(coeff_hires)
    bins = np.arange(n_regions_all.min(), n_regions_all.max() + 2) - 0.5
    axes[1, 1].hist(n_regions_all, bins=bins, color="#8172B2", rwidth=0.7)
    axes[1, 1].set_xticks(sorted(set(n_regions_all)))
    axes[1, 1].set_title("Connected high-$\\kappa$ regions\nper test sample (native $128^2$)", fontsize=11)
    axes[1, 1].set_xlabel("# connected regions"); axes[1, 1].set_ylabel("Sample count")
    fig.tight_layout()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out_dir}/fig8_eda_dataset.png", dpi=150)
    fig.savefig(f"{args.out_dir}/fig8_eda_dataset.pdf")
    plt.close(fig)

    # --- quantify "complex kappa -> high error" claim against FNO test errors ---
    correlation = None
    if Path(args.fno_eval).exists():
        with open(args.fno_eval) as f:
            fno_eval = json.load(f)
        errors = np.array(fno_eval["sample_rel_l2"])  # same order as test.npz / test_hires.npz
        if len(errors) == len(n_regions_all):
            r_regions = float(np.corrcoef(n_regions_all, errors)[0, 1])
            r_perimeter = float(np.corrcoef(perimeter_all, errors)[0, 1])
            correlation = {
                "n_test_samples": int(len(errors)),
                "pearson_r_error_vs_n_regions": r_regions,
                "pearson_r_error_vs_interface_perimeter": r_perimeter,
            }
            print("Correlation with FNO-original per-sample error:", correlation)

            fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.2))
            ax2[0].scatter(n_regions_all, errors, alpha=0.6, s=18, color="#4C72B0")
            ax2[0].set_xlabel("# connected high-$\\kappa$ regions (native $128^2$)")
            ax2[0].set_ylabel("FNO (original) test relative $L_2$ error")
            ax2[0].set_title(f"r = {r_regions:.2f}")

            ax2[1].scatter(perimeter_all, errors, alpha=0.6, s=18, color="#DD8452")
            ax2[1].set_xlabel("$\\kappa$-interface perimeter (pixel edges)")
            ax2[1].set_ylabel("FNO (original) test relative $L_2$ error")
            ax2[1].set_title(f"r = {r_perimeter:.2f}")
            fig2.suptitle("Does $\\kappa$ heterogeneity predict FNO error? (200 real test samples)")
            fig2.tight_layout()
            fig2.savefig(f"{args.out_dir}/fig9_error_vs_heterogeneity.png", dpi=150)
            fig2.savefig(f"{args.out_dir}/fig9_error_vs_heterogeneity.pdf")
            plt.close(fig2)
        else:
            print(f"WARNING: sample count mismatch ({len(errors)} errors vs "
                  f"{len(n_regions_all)} heterogeneity samples) - skipping correlation")
    else:
        print(f"WARNING: {args.fno_eval} not found - skipping error correlation")

    out = {"dataset_summary": summary, "error_vs_heterogeneity": correlation}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out_json}")
    print(f"Saved -> {args.out_dir}/fig8_eda_dataset.png, fig9_error_vs_heterogeneity.png")


if __name__ == "__main__":
    main()
