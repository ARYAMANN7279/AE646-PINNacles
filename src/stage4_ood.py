"""
Out-of-distribution (OOD) test sets for the kappa -> u operator.

Permeability maps are thresholded Gaussian random fields (GRF) on the unit square, the construction used for the
FNO/PDEBench Darcy data: a field with covariance (-Laplacian + tau^2)^(-alpha) is drawn spectrally and thresholded,
kappa = k_hi where field >= t, k_lo elsewhere. The in-distribution generator (tau, alpha, t) is tuned so that its
statistics match the PDEBench training maps (high-permeability area share, interface density); each OOD set then
changes ONE property. Labels u come from our finite-volume solver (src/solvers.py) at 128 x 128 with the boundary
treatment calibrated against PDEBench at that grid (results/stage3/solver_vs_gt.json), so every set - including the
in-distribution one - carries the same solver-vs-PDEBench discrepancy (about 0.024 relative L2). Read OOD errors
relative to the in-distribution set ("ID"), not as absolute numbers.

Run: python src/stage4_ood.py --check    (compare generator statistics with the PDEBench training maps)
     python src/stage4_ood.py --build    (write data/ood/<set>.npz with kappa128, u128 for every set)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import solvers as S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ood"
N = 128


def grf(n_maps, tau, alpha, rng, mean_factor=0.45, n=N):
    """Cosine-series (Neumann) GRF as in the FNO/PDEBench Darcy generator: coefficient (pi^2 |k|^2 + tau^2)^(-alpha/2) per
    cosine mode k >= 0. The mean mode is kept at mean_factor of its natural amplitude: this reproduces the spread of the
    high-permeability area share seen in PDEBench (with 0 every map would be close to half and half)."""
    from scipy.fft import idctn
    k = np.arange(n)
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    amp = (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2)
    amp[0, 0] *= mean_factor
    f = np.stack([idctn(amp * x, norm="ortho") for x in rng.standard_normal((n_maps, n, n))])
    return f / f.std()                                          # pooled normalisation: thresholds are in units of sd


def threshold(f, t=0.0, lo=0.1, hi=1.0, mid=None):
    if mid is None:
        return np.where(f >= t, hi, lo).astype(np.float32)
    t1, t2, kmid = mid                                         # three phases: lo | kmid | hi
    return np.where(f >= t2, hi, np.where(f >= t1, kmid, lo)).astype(np.float32)


def stats(kappa):
    hi = kappa > 0.5 * (kappa.min() + kappa.max())
    share = hi.mean(axis=(1, 2))
    iface = 0.5 * ((hi[:, 1:, :] != hi[:, :-1, :]).mean(axis=(1, 2)) + (hi[:, :, 1:] != hi[:, :, :-1]).mean(axis=(1, 2)))
    return {"share_mean": float(share.mean()), "share_sd": float(share.std()), "share_lt5pct": float((share < 0.05).mean()),
            "iface_mean": float(iface.mean()), "iface_sd": float(iface.std())}


ID = dict(tau=3.5, alpha=2.5, t=0.0)                         # tuned against PDEBench with --check
SETS = {                                                      # name: (generator changes, label)
    "ID":          ({}, "in-distribution generator"),
    "contrast_hi": ({"lo": 0.01}, "contrast 100:1 (k_lo = 0.01)"),
    "contrast_lo": ({"lo": 0.3}, "contrast 3.3:1 (k_lo = 0.3)"),
    "mostly_low":  ({"t": 0.6}, "mostly low permeability (threshold +0.6 sd)"),
    "rough":       ({"tau": 7.0}, "shorter correlation length (tau = 7)"),
    "smooth":      ({"tau": 1.75}, "longer correlation length (tau = 1.75)"),
    "three_phase": ({"mid": (-0.4, 0.4, 0.4)}, "three phases (0.1 / 0.4 / 1.0)"),
    "scaled_up":   ({"lo": 1.0, "hi": 10.0}, "both values x10 (1 / 10): same contrast, higher level"),
    "scaled_down": ({"lo": 0.01, "hi": 0.1}, "both values x0.1 (0.01 / 0.1): same contrast, lower level"),
}


def make(name, n_maps=200, seed=7):
    ch, _ = SETS[name]
    p = {**ID, **ch}
    rng = np.random.default_rng(seed)
    f = grf(n_maps, p["tau"], p["alpha"], rng)
    kap = threshold(f, p.get("t", 0.0), lo=p.get("lo", 0.1), hi=p.get("hi", 1.0), mid=p.get("mid"))
    cal = json.load(open(ROOT / "results" / "stage3" / "solver_vs_gt.json"))["per_N"]["128"]
    u = np.stack([S.solve(k, method="direct", face=cal["face"], bc=cal["bc"]) for k in kap]).astype(np.float32)
    return kap, u


def make_mixed(n_maps, seed, lo_range=(0.02, 0.5)):
    """Training data with randomised contrast: ID geometry, k_hi = 1, k_lo log-uniform in lo_range per map."""
    rng = np.random.default_rng(seed)
    f = grf(n_maps, ID["tau"], ID["alpha"], rng)
    lo = np.exp(rng.uniform(np.log(lo_range[0]), np.log(lo_range[1]), n_maps))
    kap = np.stack([np.where(fi >= 0, 1.0, l) for fi, l in zip(f, lo)]).astype(np.float32)
    cal = json.load(open(ROOT / "results" / "stage3" / "solver_vs_gt.json"))["per_N"]["128"]
    u = np.stack([S.solve(k, method="direct", face=cal["face"], bc=cal["bc"]) for k in kap]).astype(np.float32)
    return kap, u, lo


if __name__ == "__main__":
    for flag, folder, rng_lo in (("--build-mixed", "mixed", (0.02, 0.5)), ("--build-mixed-wide", "mixed_wide", (0.005, 0.5))):
        if flag in sys.argv:                                    # 900 training + 100 validation maps, randomised contrast
            (ROOT / "data" / folder).mkdir(parents=True, exist_ok=True)
            for name, n, seed in (("train", 900, 11 if folder == "mixed" else 21), ("val", 100, 12 if folder == "mixed" else 22)):
                kap, u, lo = make_mixed(n, seed, rng_lo)
                np.savez_compressed(ROOT / "data" / folder / f"{name}.npz", kappa128=kap, u128=u, k_lo=lo.astype(np.float32))
                print(folder, name, n, "k_lo range", float(lo.min()), float(lo.max()), flush=True)
    if "--check" in sys.argv:
        # reference: the 200 PDEBench test maps at 64 x 64 stored by stage3_diagnostics.py
        ref = np.load(ROOT / "results" / "stage3" / "diagnostics_arrays.npz")["kappa"]
        print("PDEBench test maps (64):", stats(ref))
        for tau in (3.0, 3.5):
            for alpha in (2.5,):
                kap = threshold(grf(600, tau, alpha, np.random.default_rng(2)))[:, ::2, ::2]
                print(f"tau {tau} alpha {alpha}:", stats(kap))
    if "--build" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
        for name in (only[0].split(",") if only else SETS):
            t0 = time.time()
            kap, u = make(name)
            np.savez_compressed(OUT / f"{name}.npz", kappa128=kap, u128=u)
            print(name, stats(kap), f"{time.time() - t0:.0f}s", flush=True)
