"""
OPTIONAL fallback: self-generate a synthetic Darcy Flow dataset that mimics
PDEBench's actual 2D Darcy setup, for use if the real PDEBench download
(src/download_data.py) is unavailable (e.g. no internet access).

This is NOT what the reported results in this project use - those come from
the real PDEBench 2D_DarcyFlow_beta1.0 dataset (src/download_data.py +
src/preprocess.py). This script is kept only as a documented, clearly-labeled
fallback permitted by the course handout ("simple Python-generated datasets
... provided the scope remains comparable and the choice is approved").

Equation: -div(kappa * grad(u)) = f, Dirichlet BC u=0 on the boundary.
Permeability kappa is piecewise-constant, kappa in {0.1, 1.0}, obtained by
thresholding a smooth Gaussian random field at zero - matching PDEBench's
actual Darcy permeability convention (NOT the continuous log-permeability
kappa=exp(coeff) used in the original Li et al. FNO paper's Darcy dataset).
"""
import numpy as np
import h5py
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from pathlib import Path
from tqdm import tqdm

LOW_PERM = 0.1
HIGH_PERM = 1.0


def generate_permeability_field(n_samples, height=64, width=64, correlation_length=0.1, seed=42):
    """
    Generate piecewise-constant permeability fields: threshold a smooth
    Gaussian random field (Matern-like spectrum) at zero, mapping to
    {LOW_PERM, HIGH_PERM} - this is PDEBench's actual Darcy convention.
    """
    rng = np.random.default_rng(seed)
    fields = []

    kx = np.fft.fftfreq(width) * width
    ky = np.fft.fftfreq(height) * height
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k_sq = KX**2 + KY**2

    l = correlation_length * max(height, width)
    spectrum = (1 + k_sq / (l**2)) ** (-2)
    filter_sqrt = np.sqrt(spectrum)

    for _ in tqdm(range(n_samples), desc="Generating permeability fields"):
        noise = rng.standard_normal((height, width))
        noise_ft = np.fft.fft2(noise)
        filtered_ft = noise_ft * filter_sqrt
        field = np.fft.ifft2(filtered_ft).real
        field = (field - field.mean()) / field.std()

        kappa = np.where(field > 0, HIGH_PERM, LOW_PERM).astype(np.float32)
        fields.append(kappa)

    return np.array(fields, dtype=np.float32)


def solve_darcy_fdm(coeff, f=1.0, height=64, width=64):
    """
    Solve -div(kappa * grad(u)) = f using finite differences (5-point stencil).
    `coeff` IS the permeability field directly (not log-permeability).
    Dirichlet BC: u = 0 on all boundaries.
    """
    N = height * width
    kappa = coeff

    dx = 1.0 / (width - 1)
    dy = 1.0 / (height - 1)

    diagonals = []
    offsets = []

    center = 2 * (kappa / dx**2 + kappa / dy**2)
    diagonals.append(center.ravel())
    offsets.append(0)

    kappa_x = (kappa[:, :-1] + kappa[:, 1:]) / 2 / dx**2
    left = np.zeros((height, width))
    left[:, 1:] = -kappa_x
    right = np.zeros((height, width))
    right[:, :-1] = -kappa_x
    diagonals.append(left.ravel())
    offsets.append(-1)
    diagonals.append(right.ravel())
    offsets.append(1)

    kappa_y = (kappa[:-1, :] + kappa[1:, :]) / 2 / dy**2
    up = np.zeros((height, width))
    up[1:, :] = -kappa_y
    down = np.zeros((height, width))
    down[:-1, :] = -kappa_y
    diagonals.append(up.ravel())
    offsets.append(-width)
    diagonals.append(down.ravel())
    offsets.append(width)

    A = diags(diagonals, offsets, shape=(N, N), format="csr")
    b = f * np.ones(N)

    boundary_mask = np.zeros((height, width), dtype=bool)
    boundary_mask[0, :] = boundary_mask[-1, :] = True
    boundary_mask[:, 0] = boundary_mask[:, -1] = True
    boundary_idx = np.where(boundary_mask.ravel())[0]

    for idx in boundary_idx:
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0

    u = spsolve(A, b)
    return u.reshape(height, width).astype(np.float32)


def generate_dataset(n_train=1000, n_test=200, height=64, width=64, seed=42):
    """Generate a synthetic fallback dataset and save as HDF5 (single train file, like PDEBench)."""
    OUT_DIR = Path(__file__).parent.parent / "data" / "raw_synthetic_fallback"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = n_train + n_test
    print(f"Generating {n_total} samples ({n_train} train-pool + {n_test} test)...")
    coeff = generate_permeability_field(n_total, height, width, seed=seed)
    tensor = np.array([
        solve_darcy_fdm(coeff[i], height=height, width=width)
        for i in tqdm(range(n_total), desc="Solving PDE")
    ])

    out_path = OUT_DIR / "2D_DarcyFlow_synthetic_fallback.hdf5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("nu", data=coeff, compression="gzip")
        f.create_dataset("tensor", data=tensor[:, None], compression="gzip")
        f.attrs["beta"] = 1.0
        f.attrs["source"] = "synthetic fallback, NOT real PDEBench data"

    print(f"\nSaved to {out_path}")
    print(f"coeff {coeff.shape}, tensor {tensor.shape}")
    return coeff, tensor


if __name__ == "__main__":
    generate_dataset(n_train=1000, n_test=200)
