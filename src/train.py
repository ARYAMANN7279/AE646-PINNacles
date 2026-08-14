"""
Training script for Darcy Flow surrogate models (MLP baseline, FNO).
Supports config files, wandb logging, checkpointing, and evaluation.
"""
import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

from models import get_model, count_parameters


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_norm_stats(data_dir):
    """Load normalization stats so metrics can be reported in physical units."""
    with open(Path(data_dir) / "norm_stats.json", "r") as f:
        stats = json.load(f)
    return stats["tensor_mean"], stats["tensor_std"]


def load_data(data_dir, batch_size, num_workers=0):
    """Load preprocessed .npz files."""
    data_dir = Path(data_dir)
    
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    test_data = np.load(data_dir / "test.npz")
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_data["inputs"]).float(),
        torch.from_numpy(train_data["targets"]).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_data["inputs"]).float(),
        torch.from_numpy(val_data["targets"]).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data["inputs"]).float(),
        torch.from_numpy(test_data["targets"]).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def rel_l2_loss(pred, target):
    """Relative L2 loss."""
    diff = pred - target
    # Flatten spatial dims for norm
    diff_flat = diff.view(diff.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    return torch.norm(diff_flat, dim=1) / torch.norm(target_flat, dim=1)


def physical_rel_l2(pred, target, tensor_mean, tensor_std):
    """
    Relative L2 error in PHYSICAL (denormalized) units.

    pred/target are standardized (zero mean, unit std over the training set).
    Denormalizing before computing the ratio matters: subtracting a constant
    changes ||target|| (but not ||pred - target||, since the constant cancels
    in the difference), so relative error computed on standardized fields is
    NOT the same number as the literature-standard physical-space relative
    error. Checkpoint selection is unaffected (same ranking either way, since
    the denominator shift is a fixed, model-independent constant per split),
    but the reported magnitude is - this metric is what should be quoted
    against literature numbers.
    """
    pred_phys = pred * tensor_std + tensor_mean
    target_phys = target * tensor_std + tensor_mean
    return rel_l2_loss(pred_phys, target_phys)


def train_epoch(model, loader, optimizer, criterion, device, tensor_mean, tensor_std, scheduler=None):
    model.train()
    total_loss = 0
    total_rel_l2 = 0
    n_batches = 0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            rel_l2 = physical_rel_l2(outputs, targets, tensor_mean, tensor_std).mean().item()
            total_rel_l2 += rel_l2
        n_batches += 1

    return total_loss / n_batches, total_rel_l2 / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, tensor_mean, tensor_std):
    model.eval()
    total_loss = 0
    total_rel_l2 = 0
    n_batches = 0

    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        rel_l2 = physical_rel_l2(outputs, targets, tensor_mean, tensor_std).mean().item()
        total_rel_l2 += rel_l2
        n_batches += 1

        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Per-sample relative L2, physical units
    sample_rel_l2 = physical_rel_l2(preds, targets, tensor_mean, tensor_std).numpy()

    return {
        "loss": total_loss / n_batches,
        "rel_l2": total_rel_l2 / n_batches,
        "sample_rel_l2": sample_rel_l2,
        "preds": preds,
        "targets": targets,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (off by default for reproducibility - no wandb login required to run this script)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader = load_data(
        config["data"]["path"],
        config["training"]["batch_size"],
        config["training"].get("num_workers", 0)
    )
    tensor_mean, tensor_std = load_norm_stats(config["data"]["path"])

    # Model
    model = get_model(config["model"]["type"], **config["model"]["params"]).to(device)
    print(f"Model: {config['model']['type']}, {count_parameters(model):,} parameters")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    
    # Scheduler
    scheduler = None
    if config["training"].get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )
    elif config["training"].get("scheduler") == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["training"]["lr"],
            epochs=config["training"]["epochs"],
            steps_per_epoch=len(train_loader)
        )
    
    # Loss
    criterion = nn.MSELoss()
    
    # Wandb
    if args.wandb:
        wandb.init(
            project="ae646-darcy-fno",
            config=config,
            name=config.get("run_name", "fno-darcy"),
        )
        wandb.watch(model, log_freq=100)
    
    # Resume
    start_epoch = 0
    best_val_rel_l2 = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_rel_l2 = checkpoint["metrics"].get("best_val_rel_l2", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_rel_l2 = train_epoch(
            model, train_loader, optimizer, criterion, device, tensor_mean, tensor_std, scheduler
        )

        val_metrics = evaluate(model, val_loader, criterion, device, tensor_mean, tensor_std)
        val_loss = val_metrics["loss"]
        val_rel_l2 = val_metrics["rel_l2"]
        
        print(f"  Train Loss: {train_loss:.6f}, Train Rel L2: {train_rel_l2:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}, Val Rel L2:   {val_rel_l2:.6f}")
        
        # Log
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rel_l2": train_rel_l2,
                "val_loss": val_loss,
                "val_rel_l2": val_rel_l2,
                "lr": optimizer.param_groups[0]["lr"],
            })
        
        # Save best
        if val_rel_l2 < best_val_rel_l2:
            best_val_rel_l2 = val_rel_l2
            save_checkpoint(
                model, optimizer, scheduler, epoch, config,
                {"best_val_rel_l2": best_val_rel_l2},
                results_dir / "best_model.pt"
            )
            print(f"  ✓ New best model saved (val_rel_l2={val_rel_l2:.6f})")

        # Save a periodic checkpoint (only for resume capability - kept infrequent
        # to avoid multi-GB checkpoint bloat; best_model.pt is what's actually used)
        save_every = config["output"].get("save_every", 0)
        if save_every and (epoch + 1) % save_every == 0 and (epoch + 1) != config["training"]["epochs"]:
            save_checkpoint(
                model, optimizer, scheduler, epoch, config,
                {"val_rel_l2": val_rel_l2},
                results_dir / "last_checkpoint.pt"
            )

    # Final evaluation on test set
    print("\n=== Final Test Evaluation (physical units) ===")
    best_checkpoint = torch.load(results_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, tensor_mean, tensor_std)
    test_rel_l2 = test_metrics["rel_l2"]
    test_loss = test_metrics["loss"]
    sample_rel_l2 = test_metrics["sample_rel_l2"]
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Rel L2: {test_rel_l2:.6f}")
    print(f"Test Rel L2 (per sample): mean={sample_rel_l2.mean():.6f}, "
          f"std={sample_rel_l2.std():.6f}, "
          f"min={sample_rel_l2.min():.6f}, max={sample_rel_l2.max():.6f}")
    
    # Save test metrics
    test_results = {
        "test_loss": float(test_loss),
        "test_rel_l2": float(test_rel_l2),
        "sample_rel_l2_mean": float(sample_rel_l2.mean()),
        "sample_rel_l2_std": float(sample_rel_l2.std()),
        "sample_rel_l2_min": float(sample_rel_l2.min()),
        "sample_rel_l2_max": float(sample_rel_l2.max()),
        "best_epoch": best_checkpoint["epoch"],
    }
    
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    if args.wandb:
        wandb.log({"test_loss": test_loss, "test_rel_l2": test_rel_l2})
        wandb.finish()
    
    print(f"\nDone! Results saved to {results_dir}")


if __name__ == "__main__":
    main()