"""
Evaluation and visualization script for trained models.
Generates plots, computes metrics, and creates comparison tables.
"""
import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from models import get_model, count_parameters


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_dir):
    data_dir = Path(data_dir)
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    test_data = np.load(data_dir / "test.npz")
    return train_data, val_data, test_data


def load_model(checkpoint_path, config, device):
    model = get_model(config["model"]["type"], **config["model"]["params"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_norm_stats(data_dir):
    with open(Path(data_dir) / "norm_stats.json", "r") as f:
        stats = json.load(f)
    return stats["tensor_mean"], stats["tensor_std"]


def rel_l2(pred, target):
    """Relative L2 error per sample (on whatever units pred/target are already in)."""
    diff = pred - target
    diff_flat = diff.view(diff.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    return torch.norm(diff_flat, dim=1) / torch.norm(target_flat, dim=1)


def mse(pred, target):
    return torch.mean((pred - target) ** 2, dim=(1,2,3))


@torch.no_grad()
def evaluate_model(model, data_loader, device, tensor_mean, tensor_std):
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())
        all_inputs.append(inputs.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    inputs = torch.cat(all_inputs, dim=0)

    # Denormalize to physical units before computing error - matches the
    # literature-standard relative L2 metric (see src/train.py:physical_rel_l2
    # for why normalized-space error is not the same number).
    preds_phys = preds * tensor_std + tensor_mean
    targets_phys = targets * tensor_std + tensor_mean

    # Per-sample metrics, physical units
    sample_rel_l2 = rel_l2(preds_phys, targets_phys).numpy()
    sample_mse = mse(preds_phys, targets_phys).numpy()

    return {
        "preds": preds.numpy(),
        "targets": targets.numpy(),
        "preds_phys": preds_phys.numpy(),
        "targets_phys": targets_phys.numpy(),
        "inputs": inputs.numpy(),
        "sample_rel_l2": sample_rel_l2,
        "sample_mse": sample_mse,
        "mean_rel_l2": sample_rel_l2.mean(),
        "std_rel_l2": sample_rel_l2.std(),
        "median_rel_l2": np.median(sample_rel_l2),
    }


def plot_samples(inputs, targets, preds, indices, save_path, coeff_mean=0.0, coeff_std=1.0):
    """Plot input permeability, target pressure, predicted pressure, and error.

    `inputs` is the normalized model input; `targets`/`preds` are expected in
    PHYSICAL units already (pass preds_phys/targets_phys) so the plots and the
    reported error are consistent with each other.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        coeff = inputs[idx, ..., 0] * coeff_std + coeff_mean  # denormalized permeability
        target = targets[idx, ..., 0]
        pred = preds[idx, ..., 0]
        error = np.abs(pred - target)
        
        vmin, vmax = target.min(), target.max()
        err_max = error.max()
        
        im0 = axes[i, 0].imshow(coeff, cmap="viridis", origin="lower")
        axes[i, 0].set_title(f"Input: Permeability (sample {idx})")
        axes[i, 0].axis("off")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        im1 = axes[i, 1].imshow(target, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Target: Pressure")
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        im2 = axes[i, 2].imshow(pred, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Prediction: Pressure")
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        im3 = axes[i, 3].imshow(error, cmap="hot", origin="lower", vmin=0, vmax=err_max)
        axes[i, 3].set_title(f"Absolute Error (max={err_max:.3f})")
        axes[i, 3].axis("off")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(metrics, save_path, model_name="Model"):
    """Plot histogram of relative L2 errors."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(metrics["sample_rel_l2"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(metrics["mean_rel_l2"], color="red", linestyle="--", 
                    label=f"Mean: {metrics['mean_rel_l2']:.4f}")
    axes[0].axvline(metrics["median_rel_l2"], color="green", linestyle="--",
                    label=f"Median: {metrics['median_rel_l2']:.4f}")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{model_name}: Error Distribution")
    axes[0].legend()
    axes[0].set_yscale("log")
    
    axes[1].boxplot(metrics["sample_rel_l2"], orientation="vertical")
    axes[1].set_ylabel("Relative L2 Error")
    axes[1].set_title(f"{model_name}: Box Plot")
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(results_dir, save_path):
    """Plot training curves from wandb or saved metrics."""
    # Check for test_metrics.json
    metrics_file = Path(results_dir) / "test_metrics.json"
    if not metrics_file.exists():
        return
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # If we have checkpoint with history, we could plot more
    # For now just print final metrics
    print(f"Final test metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to visualize")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(config["output"]["results_dir"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(config["data"]["path"])
    tensor_mean, tensor_std = load_norm_stats(config["data"]["path"])
    with open(Path(config["data"]["path"]) / "norm_stats.json", "r") as f:
        norm_stats = json.load(f)
    coeff_mean, coeff_std = norm_stats["coeff_mean"], norm_stats["coeff_std"]

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_data["inputs"]).float(),
        torch.from_numpy(test_data["targets"]).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Evaluate (physical units)
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device, tensor_mean, tensor_std)

    print(f"\nTest Set Results (physical units):")
    print(f"  Mean Rel L2:  {metrics['mean_rel_l2']:.6f}")
    print(f"  Std Rel L2:   {metrics['std_rel_l2']:.6f}")
    print(f"  Median Rel L2: {metrics['median_rel_l2']:.6f}")
    print(f"  Min Rel L2:   {metrics['sample_rel_l2'].min():.6f}")
    print(f"  Max Rel L2:   {metrics['sample_rel_l2'].max():.6f}")
    
    # Save metrics
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump({
            "mean_rel_l2": float(metrics["mean_rel_l2"]),
            "std_rel_l2": float(metrics["std_rel_l2"]),
            "median_rel_l2": float(metrics["median_rel_l2"]),
            "min_rel_l2": float(metrics["sample_rel_l2"].min()),
            "max_rel_l2": float(metrics["sample_rel_l2"].max()),
            "sample_rel_l2": metrics["sample_rel_l2"].tolist(),
        }, f, indent=2)
    
    # Visualize samples (best, median, worst)
    rel_l2 = metrics["sample_rel_l2"]
    best_idx = np.argmin(rel_l2)
    worst_idx = np.argmax(rel_l2)
    median_idx = np.argsort(rel_l2)[len(rel_l2)//2]
    
    # Random samples
    random_indices = np.random.choice(len(rel_l2), min(args.n_samples - 3, len(rel_l2) - 3), replace=False)
    vis_indices = [best_idx, median_idx, worst_idx] + list(random_indices)
    
    print(f"\nVisualizing samples: {vis_indices}")
    plot_samples(
        metrics["inputs"], metrics["targets_phys"], metrics["preds_phys"],
        vis_indices, out_dir / "sample_predictions.png",
        coeff_mean=coeff_mean, coeff_std=coeff_std,
    )
    
    plot_error_distribution(metrics, out_dir / "error_distribution.png", 
                           config["model"]["type"].upper())
    
    print(f"\nEvaluation complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()