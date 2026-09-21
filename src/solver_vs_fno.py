"""
Classical finite-volume solver vs the PDEBench ground truth (and vs the FNO), at N = 32 / 64 / 128.

1. Calibrate the boundary-distance parameter `bc` and face averaging on the VALIDATION split (per N).
2. Report relative L2 error vs the ground truth on the TEST split with the calibrated setting
   (and with the strict cell-centred setting bc = 0.5 for reference).
3. Time the solvers (single CPU thread, median per solve): sparse direct, and AMG-preconditioned CG if pyamg exists.

Coarse grids use stride sub-sampling of both kappa and u, exactly what the FNO sees (see stage3_train.Data).
Writes results/stage3/solver_vs_gt.json.   Run: OMP_NUM_THREADS=1 python src/solver_vs_fno.py
"""
import json
import os
from pathlib import Path

import numpy as np

import solvers as S

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "data" / "cache"


def rel_l2(p, t):
    return float(np.mean(np.linalg.norm((p - t).reshape(len(p), -1), axis=1) / np.linalg.norm(t.reshape(len(t), -1), axis=1)))


def run_set(K, U, idx, n, face, bc, method="direct"):
    P = np.stack([S.solve(S.coarsen(K[i], n), method=method, face=face, bc=bc) for i in idx])
    T = np.stack([S.coarsen(U[i], n) for i in idx])
    return rel_l2(P, T)


def main():
    K = np.load(C / "kappa128.npy", mmap_mode="r"); U = np.load(C / "u128.npy", mmap_mode="r")
    sp = np.load(C / "splits.npz")
    val, test = sp["val"], sp["test"]
    try:
        import pyamg  # noqa
        have_amg = True
    except ImportError:
        have_amg = False
    out = {"have_pyamg": have_amg, "omp_threads": os.environ.get("OMP_NUM_THREADS"), "per_N": {}}
    for n in (32, 64, 128):
        cal = {}
        for face in ("harmonic", "arithmetic"):
            for bc in (0.5, 0.75, 1.0, 1.25, 1.5):
                cal[f"{face}|{bc}"] = run_set(K, U, val[:40], n, face, bc)
        best = min(cal, key=cal.get)
        face, bc = best.split("|"); bc = float(bc)
        # refine bc around the best with the same face
        fine = {b: run_set(K, U, val[:40], n, face, b) for b in np.round(np.linspace(max(0.5, bc - 0.25), bc + 0.25, 6), 3)}
        bc = float(min(fine, key=fine.get))
        strict = run_set(K, U, test, n, "harmonic", 0.5)
        calib = run_set(K, U, test, n, face, bc)
        kt = [S.coarsen(K[i], n) for i in test[:20]]
        t_dir = S.time_solver(kt, "direct", face, bc, repeats=2)
        t_amg = S.time_solver(kt, "cg_amg", face, bc, repeats=2) if have_amg else None
        ref = S.solve(kt[0], face=face, bc=bc)
        amg_err = float(np.linalg.norm(S.solve(kt[0], method="cg_amg", face=face, bc=bc) - ref) / np.linalg.norm(ref)) if have_amg else None
        out["per_N"][n] = {"val_grid": cal, "val_refine_bc": {str(k): v for k, v in fine.items()}, "face": face, "bc": bc,
                           "test_err_strict_fv": strict, "test_err_calibrated": calib,
                           "ms_direct": 1e3 * t_dir, "ms_cg_amg": None if t_amg is None else 1e3 * t_amg, "amg_vs_direct_relerr": amg_err}
        print(n, {k: v for k, v in out["per_N"][n].items() if k not in ("val_grid", "val_refine_bc")}, flush=True)
    (ROOT / "results" / "stage3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "stage3" / "solver_vs_gt.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
