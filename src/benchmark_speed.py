"""
Real inference-speed benchmark: trained FNO vs trained MLP vs a numerical
FDM Darcy solver, all measured on this machine (not asserted numbers).

Usage:
    python src/benchmark_speed.py \
        --fno-config configs/fno.yaml --fno-checkpoint results/run_001/best_model.pt \
        --mlp-config configs/mlp.yaml --mlp-checkpoint results/run_002/best_model.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from models import get_model
from generate_data import solve_darcy_fdm


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def time_model(model, inputs, device, n_warmup=5, n_repeat=50):
    """Time single-sample forward passes (median over n_repeat runs, after warmup)."""
    model.eval()
    x = inputs[:1].to(device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000  # ms


def time_fdm_solver(coeff_samples, n_repeat=20):
    """Time the scipy sparse FDM solve (src/generate_data.py:solve_darcy_fdm), on CPU."""
    times = []
    H, W = coeff_samples.shape[1:]
    for i in range(min(n_repeat, len(coeff_samples))):
        t0 = time.perf_counter()
        solve_darcy_fdm(coeff_samples[i], height=H, width=W)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fno-config", default="configs/fno.yaml")
    parser.add_argument("--fno-checkpoint", default="results/run_001/best_model.pt")
    parser.add_argument("--mlp-config", default="configs/mlp.yaml")
    parser.add_argument("--mlp-checkpoint", default="results/run_002/best_model.pt")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/benchmark_speed.json")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural network inference device: {device}")
    print("FDM solver device: cpu (scipy sparse direct solve - not GPU-accelerated by design)")

    test_data = np.load(Path(args.data_dir) / "test.npz")
    inputs = torch.from_numpy(test_data["inputs"]).float()

    results = {"device": str(device)}

    # FNO
    fno_cfg = load_config(args.fno_config)
    fno = get_model(fno_cfg["model"]["type"], **fno_cfg["model"]["params"]).to(device)
    ckpt = torch.load(args.fno_checkpoint, map_location=device)
    fno.load_state_dict(ckpt["model_state_dict"])
    fno_ms = time_model(fno, inputs, device)
    results["fno_ms_per_sample"] = fno_ms
    print(f"FNO:  {fno_ms:.3f} ms/sample (median of 50 runs, {device})")

    # MLP
    mlp_cfg = load_config(args.mlp_config)
    mlp = get_model(mlp_cfg["model"]["type"], **mlp_cfg["model"]["params"]).to(device)
    ckpt = torch.load(args.mlp_checkpoint, map_location=device)
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp_ms = time_model(mlp, inputs, device)
    results["mlp_ms_per_sample"] = mlp_ms
    print(f"MLP:  {mlp_ms:.3f} ms/sample (median of 50 runs, {device})")

    # FDM solver, on the same permeability fields (denormalized to physical units)
    with open(Path(args.data_dir) / "norm_stats.json") as f:
        stats = json.load(f)
    coeff_phys = test_data["inputs"][..., 0] * stats["coeff_std"] + stats["coeff_mean"]
    fdm_ms = time_fdm_solver(coeff_phys)
    results["fdm_ms_per_sample"] = fdm_ms
    print(f"FDM solver (scipy sparse, cpu): {fdm_ms:.3f} ms/sample (median of 20 runs)")

    results["fno_speedup_vs_fdm"] = fdm_ms / fno_ms
    results["mlp_speedup_vs_fdm"] = fdm_ms / mlp_ms
    print(f"\nFNO is {results['fno_speedup_vs_fdm']:.1f}x faster than the FDM solver")
    print(f"MLP is {results['mlp_speedup_vs_fdm']:.1f}x faster than the FDM solver")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
