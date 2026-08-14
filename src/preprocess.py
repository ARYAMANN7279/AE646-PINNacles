"""
Preprocess the real PDEBench 2D Darcy Flow (beta=1.0) dataset.

The raw file (data/raw_pdebench/2D_DarcyFlow_beta1.0_Train.hdf5) holds all
10,000 PDEBench samples at native 128x128 resolution:
  - "nu":     (10000, 128, 128) piecewise-constant permeability field, values in {0.1, 1.0}
  - "tensor": (10000, 1, 128, 128) steady-state pressure solution

This script:
1. Draws a reproducible subset (seed=42): 1000 samples for train/val, 200 held
   out for test - matching the sample counts used in the FNO literature
   (Li et al. 2021) so results are directly comparable.
2. Downsamples 128x128 -> 64x64 by taking every 2nd grid point (a clean,
   literature-standard subsampling; PDEBench's own loaders do the same).
3. Splits the 1000-sample pool into 900 train / 100 val.
4. Normalizes (permeability, pressure) using TRAIN statistics only.
5. Adds coordinate channels: input = [permeability, x_coord, y_coord].
6. Also saves the 200 test samples' NATIVE 128x128 fields (unnormalized,
   raw physical units) as test_hires.npz, so a trained 64x64 model's
   zero-shot super-resolution behaviour can be checked against real
   PDEBench ground truth (not synthetic/fabricated data).
"""
import json
from pathlib import Path

import h5py
import numpy as np

RAW_FILE = Path(__file__).parent.parent / "data" / "raw_pdebench" / "2D_DarcyFlow_beta1.0_Train.hdf5"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

N_TRAINVAL = 1000
N_TEST = 200
SEED = 42


def load_raw(filepath: Path):
    with h5py.File(filepath, "r") as f:
        nu = f["nu"][:]                 # (10000, 128, 128)
        tensor = f["tensor"][:, 0]      # (10000, 128, 128) - squeeze channel dim
        x = f["x-coordinate"][:]        # (128,)
        y = f["y-coordinate"][:]        # (128,)
    return nu, tensor, x, y


def downsample(field: np.ndarray, stride: int = 2) -> np.ndarray:
    """Subsample a (N, H, W) field by taking every `stride`-th grid point."""
    return field[:, ::stride, ::stride]


def normalize_data(train_coeff, train_tensor, val_coeff, val_tensor, test_coeff, test_tensor):
    """Normalize using train statistics only (global scalar mean/std)."""
    coeff_mean, coeff_std = train_coeff.mean(), train_coeff.std()
    tensor_mean, tensor_std = train_tensor.mean(), train_tensor.std()

    def norm_c(x):
        return (x - coeff_mean) / coeff_std

    def norm_t(x):
        return (x - tensor_mean) / tensor_std

    stats = {
        "coeff_mean": float(coeff_mean),
        "coeff_std": float(coeff_std),
        "tensor_mean": float(tensor_mean),
        "tensor_std": float(tensor_std),
    }
    return (
        norm_c(train_coeff), norm_t(train_tensor),
        norm_c(val_coeff), norm_t(val_tensor),
        norm_c(test_coeff), norm_t(test_tensor),
        stats,
    )


def add_coordinates(coeff: np.ndarray, tensor: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Add coordinate channels to input.
    coeff: (N, H, W) -> inputs: (N, H, W, 3) with [coeff, x_coord, y_coord]
    tensor: (N, H, W) -> targets: (N, H, W, 1)
    """
    N, H, W = coeff.shape
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = np.broadcast_to(X, (N, H, W))
    Y = np.broadcast_to(Y, (N, H, W))
    inputs = np.stack([coeff, X, Y], axis=-1).astype(np.float32)
    targets = tensor[..., np.newaxis].astype(np.float32)
    return inputs, targets


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw PDEBench data from {RAW_FILE} ...")
    nu, tensor, x, y = load_raw(RAW_FILE)
    print(f"Raw: nu {nu.shape}, tensor {tensor.shape}, grid {x.shape[0]}x{y.shape[0]}")

    # Reproducible non-overlapping subset: first N_TRAINVAL for train/val, next N_TEST for test
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(nu.shape[0])
    trainval_idx = perm[:N_TRAINVAL]
    test_idx = perm[N_TRAINVAL:N_TRAINVAL + N_TEST]

    nu_trainval, tensor_trainval = nu[trainval_idx], tensor[trainval_idx]
    nu_test_hires, tensor_test_hires = nu[test_idx], tensor[test_idx]

    # Downsample 128x128 -> 64x64 (train/val use downsampled only)
    nu_trainval_ds = downsample(nu_trainval)
    tensor_trainval_ds = downsample(tensor_trainval)
    nu_test_ds = downsample(nu_test_hires)
    tensor_test_ds = downsample(tensor_test_hires)
    x_ds, y_ds = x[::2], y[::2]

    # Split train/val pool 900/100 (seed=42)
    split_rng = np.random.default_rng(SEED)
    split_perm = split_rng.permutation(N_TRAINVAL)
    train_idx, val_idx = split_perm[:900], split_perm[900:]

    train_coeff, train_tensor = nu_trainval_ds[train_idx], tensor_trainval_ds[train_idx]
    val_coeff, val_tensor = nu_trainval_ds[val_idx], tensor_trainval_ds[val_idx]
    test_coeff, test_tensor = nu_test_ds, tensor_test_ds

    print(f"Split (64x64): train {len(train_coeff)}, val {len(val_coeff)}, test {len(test_coeff)}")

    # Normalize using train stats only
    print("Normalizing (train statistics only)...")
    (train_coeff_n, train_tensor_n,
     val_coeff_n, val_tensor_n,
     test_coeff_n, test_tensor_n,
     stats) = normalize_data(
        train_coeff, train_tensor, val_coeff, val_tensor, test_coeff, test_tensor
    )

    # Add coordinate channels
    train_inputs, train_targets = add_coordinates(train_coeff_n, train_tensor_n, x_ds, y_ds)
    val_inputs, val_targets = add_coordinates(val_coeff_n, val_tensor_n, x_ds, y_ds)
    test_inputs, test_targets = add_coordinates(test_coeff_n, test_tensor_n, x_ds, y_ds)

    print(f"Train inputs: {train_inputs.shape}, targets: {train_targets.shape}")
    print(f"Val inputs:   {val_inputs.shape}, targets: {val_targets.shape}")
    print(f"Test inputs:  {test_inputs.shape}, targets: {test_targets.shape}")

    np.savez_compressed(PROCESSED_DIR / "train.npz", inputs=train_inputs, targets=train_targets)
    np.savez_compressed(PROCESSED_DIR / "val.npz", inputs=val_inputs, targets=val_targets)
    np.savez_compressed(PROCESSED_DIR / "test.npz", inputs=test_inputs, targets=test_targets)

    with open(PROCESSED_DIR / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save NATIVE 128x128 test fields (raw physical units, un-normalized) for the
    # zero-shot super-resolution check - real PDEBench ground truth, not fabricated.
    np.savez_compressed(
        PROCESSED_DIR / "test_hires.npz",
        coeff=nu_test_hires.astype(np.float32),
        tensor=tensor_test_hires.astype(np.float32),
        x=x.astype(np.float32),
        y=y.astype(np.float32),
    )

    print(f"\nNormalization stats: {stats}")
    print(f"Done. Processed data saved to {PROCESSED_DIR}")
    print("Next step: run training with `python src/train.py --config configs/fno.yaml`")


if __name__ == "__main__":
    main()
