"""
Cache the raw PDEBench Darcy file as memory-mappable .npy arrays plus the exact split indices.

Why: the Stage 3 experiments launch many training jobs in parallel; each one reads the arrays
through np.load(..., mmap_mode="r") instead of re-reading the 1.3 GB HDF5 file.

The split indices are re-derived with exactly the same logic as src/preprocess.py (seed 42):
  perm         = rng(42).permutation(10000)
  trainval     = perm[:1000]          -> split_perm = rng(42).permutation(1000): train = first 900, val = last 100
  test         = perm[1000:1200]      (the 200 held-out test samples, never used for training/selection)
  extra        = perm[1200:]          (8800 further real samples, used ONLY by the data-scaling study)

Writes data/cache/{kappa128.npy, u128.npy, splits.npz} and verifies that the derived train/val/test
arrays equal data/processed/*.npz (when present).

Run: python src/data_cache.py
"""
import json
from pathlib import Path

import numpy as np

import preprocess as pp

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def build_splits(n_total=10000):
    rng = np.random.default_rng(pp.SEED)
    perm = rng.permutation(n_total)
    trainval = perm[:pp.N_TRAINVAL]
    test = perm[pp.N_TRAINVAL:pp.N_TRAINVAL + pp.N_TEST]
    extra = perm[pp.N_TRAINVAL + pp.N_TEST:]
    split_perm = np.random.default_rng(pp.SEED).permutation(pp.N_TRAINVAL)
    train, val = trainval[split_perm[:900]], trainval[split_perm[900:]]
    return {"train": train, "val": val, "test": test, "extra": extra}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    nu, tensor, x, y = pp.load_raw(pp.RAW_FILE)
    np.save(CACHE / "kappa128.npy", nu.astype(np.float32))
    np.save(CACHE / "u128.npy", tensor.astype(np.float32))
    splits = build_splits(nu.shape[0])
    np.savez(CACHE / "splits.npz", x=x, y=y, **splits)
    print({k: len(v) for k, v in splits.items()}, "| grid", x.shape, "coords range", float(x.min()), float(x.max()))

    # verify against the processed arrays used for the Stage 2 / reported runs
    proc = pp.PROCESSED_DIR
    if (proc / "train.npz").exists():
        with open(proc / "norm_stats.json") as f:
            st = json.load(f)
        for name in ("train", "val", "test"):
            d = np.load(proc / f"{name}.npz")
            idx = splits[name]
            k = (nu[idx][:, ::2, ::2] - st["coeff_mean"]) / st["coeff_std"]
            u = (tensor[idx][:, ::2, ::2] - st["tensor_mean"]) / st["tensor_std"]
            ok = np.allclose(k, d["inputs"][..., 0], atol=1e-5) and np.allclose(u, d["targets"][..., 0], atol=1e-5)
            print(f"  {name}: derived split == data/processed/{name}.npz : {ok}")
            assert ok, f"split mismatch for {name}"


if __name__ == "__main__":
    main()
