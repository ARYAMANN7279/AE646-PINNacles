"""
Real ablation over MLP baseline design choices: depth and optimizer.

Motivation: the final baseline (3x2048 hidden layers, AdamW) wasn't the only
thing tried - this script actually trains the shallower/worse variants and
reports their real numbers, so the choice of final baseline is backed by
evidence rather than asserted. Every number here comes from a real training
run on the same real PDEBench data/split used everywhere else in this repo
(no synthetic data, no hand-picked results).

Run: python src/ablation_mlp.py --config configs/mlp.yaml
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model, count_parameters
from train import load_data, load_norm_stats, train_epoch, evaluate

# (name, hidden_dims, optimizer_type, lr) - the 3x2048/AdamW row matches the
# reported final baseline (configs/mlp.yaml) and is included as a sanity check.
CONFIGS = [
    ("1-layer (2048)",        [2048],             "adamw", 1e-3),
    ("2-layer (2048x2)",      [2048, 2048],       "adamw", 1e-3),
    ("3-layer (2048x3, final)", [2048, 2048, 2048], "adamw", 1e-3),
    ("3-layer, SGD+momentum", [2048, 2048, 2048], "sgd",   1e-2),
]


def make_optimizer(kind, params, lr, weight_decay):
    if kind == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif kind == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(kind)


def run_one(name, hidden_dims, opt_kind, lr, train_loader, val_loader, test_loader,
            tensor_mean, tensor_std, device, epochs, weight_decay=1e-4):
    print(f"\n=== Ablation config: {name} ===")
    model = get_model("mlp", input_channels=3, output_channels=1,
                       height=64, width=64, hidden_dims=hidden_dims).to(device)
    n_params = count_parameters(model)
    optimizer = make_optimizer(opt_kind, model.parameters(), lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device,
                    tensor_mean, tensor_std, scheduler)
        val_metrics = evaluate(model, val_loader, criterion, device, tensor_mean, tensor_std)
        if val_metrics["rel_l2"] < best_val:
            best_val = val_metrics["rel_l2"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/{epochs}  val_rel_l2={val_metrics['rel_l2']:.4f}  "
                  f"(best={best_val:.4f} @ {best_epoch+1})")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device, tensor_mean, tensor_std)
    sample = test_metrics["sample_rel_l2"]
    result = {
        "name": name,
        "hidden_dims": hidden_dims,
        "optimizer": opt_kind,
        "lr": lr,
        "params": n_params,
        "best_epoch": best_epoch,
        "test_rel_l2_mean": float(sample.mean()),
        "test_rel_l2_median": float(np.median(sample)),
        "test_rel_l2_std": float(sample.std()),
        "test_rel_l2_min": float(sample.min()),
        "test_rel_l2_max": float(sample.max()),
    }
    print(f"  -> test mean rel L2 = {result['test_rel_l2_mean']:.4f} "
          f"({n_params:,} params, best epoch {best_epoch+1})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mlp.yaml",
                     help="Used only for data path / batch size / epoch budget")
    ap.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    ap.add_argument("--output", default="results/ablation_mlp.json")
    ap.add_argument("--figure", default="results/figures/fig7_mlp_ablation.png")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    data_dir = cfg["data"]["path"]

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = load_data(data_dir, batch_size)
    tensor_mean, tensor_std = load_norm_stats(data_dir)

    results = []
    for name, hidden_dims, opt_kind, lr in CONFIGS:
        results.append(run_one(name, hidden_dims, opt_kind, lr,
                                train_loader, val_loader, test_loader,
                                tensor_mean, tensor_std, device, epochs))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")

    # Figure: bar chart of mean rel L2 per config, final baseline highlighted
    names = [r["name"] for r in results]
    means = [r["test_rel_l2_mean"] for r in results]
    stds = [r["test_rel_l2_std"] for r in results]
    colors = ["#4C72B0" if "final" not in n else "#DD8452" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test mean relative $L_2$ error")
    ax.set_title("MLP baseline ablation: depth and optimizer (real runs, 200 test samples)")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.002, f"{m:.4f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    fig.savefig(str(args.figure).replace(".png", ".pdf"))
    print(f"Saved -> {args.figure}")


if __name__ == "__main__":
    main()
