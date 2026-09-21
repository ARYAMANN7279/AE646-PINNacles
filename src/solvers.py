"""
Classical reference solvers for the Darcy problem   -div(kappa grad u) = f  on (0,1)^2,  u = 0 on the boundary.

Cell-centred finite-volume discretisation on an N x N grid (h = 1/N, unknowns at cell centres, which is the
grid PDEBench's arrays use: x_i = (i + 1/2) / N). Face conductivities are averaged from the two adjacent cells
("harmonic" is the standard conservative choice, "arithmetic" is also available); faces on the domain boundary
use the half-cell distance to the Dirichlet value.

The matrix is assembled in one vectorised COO pass (no Python loops), which is what a competent implementation
does; the older `generate_data.solve_darcy_fdm` sets its boundary rows in a Python loop over CSR rows and is
orders of magnitude slower, so it must NOT be used as the timing reference.

Solvers:
  method="direct"  : scipy.sparse.linalg.spsolve (sparse LU, SuperLU/UMFPACK)
  method="cg_amg"  : conjugate gradients with an algebraic-multigrid preconditioner (pyamg, if installed)
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve


def assemble(kappa, f=1.0, face="harmonic", bc=0.5):
    """Return (A, b) for the N x N FV/FD system (CSR matrix, rhs vector).

    `bc` is the distance, in grid cells, from the first/last array row to the Dirichlet boundary
    (u = 0). bc = 0.5 is the strict cell-centred finite-volume convention; bc = 1 puts the boundary one
    full cell outside the array (node-centred convention). The PDEBench ground truth behaves like
    bc ~ 1, so `bc` is calibrated on validation data (see src/solver_vs_fno.py)."""
    n = kappa.shape[0]
    assert kappa.shape == (n, n)
    h2 = (1.0 / n) ** 2
    k = kappa.astype(np.float64)

    def face_avg(a, b):
        if face == "harmonic":
            return 2.0 * a * b / (a + b)
        return 0.5 * (a + b)

    idx = np.arange(n * n).reshape(n, n)
    kx = face_avg(k[:, :-1], k[:, 1:]) / h2          # faces between (i, j) and (i, j+1)
    ky = face_avg(k[:-1, :], k[1:, :]) / h2          # faces between (i, j) and (i+1, j)

    diag = np.zeros((n, n))
    diag[:, :-1] += kx
    diag[:, 1:] += kx
    diag[:-1, :] += ky
    diag[1:, :] += ky
    # boundary faces: flux = kappa * (u_cell - 0) / (bc * h)  ->  kappa / (bc * h^2) on the diagonal
    cb = 1.0 / bc
    diag[:, 0] += cb * k[:, 0] / h2
    diag[:, -1] += cb * k[:, -1] / h2
    diag[0, :] += cb * k[0, :] / h2
    diag[-1, :] += cb * k[-1, :] / h2

    rows = np.concatenate([idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()])
    cols = np.concatenate([idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()])
    vals = np.concatenate([diag.ravel(), -kx.ravel(), -kx.ravel(), -ky.ravel(), -ky.ravel()])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    b = np.full(n * n, float(f))
    return A, b


def solve(kappa, f=1.0, method="direct", face="harmonic", bc=0.5, tol=1e-8):
    """Solve for u (N x N, float32). `tol` is the relative residual tolerance for the iterative solver."""
    n = kappa.shape[0]
    A, b = assemble(kappa, f, face, bc)
    if method == "direct":
        u = spsolve(A.tocsc(), b)
    elif method == "cg_amg":
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=50)
        u, info = cg(A, b, rtol=tol, M=ml.aspreconditioner(cycle="V"), maxiter=200)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
    else:
        raise ValueError(method)
    return u.reshape(n, n).astype(np.float32)


def residual(kappa, u, f=1.0, face="harmonic", bc=0.5):
    """Relative residual ||A u - b|| / ||b|| of a candidate solution under the FV operator."""
    A, b = assemble(kappa, f, face, bc)
    r = A @ u.astype(np.float64).ravel() - b
    return float(np.linalg.norm(r) / np.linalg.norm(b))


def coarsen(field, n_out, mode="stride"):
    """Reduce an (N, N) field to (n_out, n_out): 'stride' takes every (N/n_out)-th cell, 'mean' block-averages."""
    n = field.shape[0]
    s = n // n_out
    assert n % n_out == 0
    if mode == "stride":
        return field[::s, ::s]
    return field.reshape(n_out, s, n_out, s).mean(axis=(1, 3))


def time_solver(kappas, method="direct", face="harmonic", bc=0.5, repeats=1):
    """Median wall-clock seconds per solve over the given fields (single process, CPU)."""
    times = []
    for k in kappas:
        for _ in range(repeats):
            t0 = time.perf_counter()
            solve(k, method=method, face=face, bc=bc)
            times.append(time.perf_counter() - t0)
    return float(np.median(times))
