"""Flux and mass-conservation audit of Darcy surrogates (PDEBench 2D Darcy test split, 64x64).
Pressure rel-L2 is what papers report. Here we also score the Darcy flux q = -kappa grad u (two-point face
fluxes with harmonic face permeability, Dirichlet u=0 half a cell outside the array), the share of flux error
that sits near phase interfaces, the local mass-balance residual div q - f, and the total boundary outflow,
which must equal the integrated source for any conservative field. Labels are scored the same way, as a floor."""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; N = 64; h = 1.0 / N

def faces(u, k, bc=0.5):
    """Outward-positive face fluxes per unit length: interior x/y faces plus the four boundary sides."""
    hx = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1]); hy = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
    qx = -hx * (u[:, 1:] - u[:, :-1]) / h; qy = -hy * (u[1:] - u[:-1]) / h
    bL = k[:, 0] * u[:, 0] / (bc * h); bR = k[:, -1] * u[:, -1] / (bc * h)    # flux leaving through each side
    bB = k[0] * u[0] / (bc * h); bT = k[-1] * u[-1] / (bc * h)
    return qx, qy, (bL, bR, bB, bT)

def div(qx, qy, b):
    d = np.zeros((N, N)); d[:, :-1] += qx; d[:, 1:] -= qx; d[:-1] += qy; d[1:] -= qy
    d[:, 0] += b[0]; d[:, -1] += b[1]; d[0] += b[2]; d[-1] += b[3]
    return d * h                                     # net outflow per cell; equals f*h^2 = h^2 when conservative

def band(k, w=2):
    e = np.zeros_like(k, bool); e[:, 1:] |= k[:, 1:] != k[:, :-1]; e[:, :-1] |= k[:, 1:] != k[:, :-1]
    e[1:] |= k[1:] != k[:-1]; e[:-1] |= k[1:] != k[:-1]
    for _ in range(w - 1):
        f = e.copy(); f[1:] |= e[:-1]; f[:-1] |= e[1:]; f[:, 1:] |= e[:, :-1]; f[:, :-1] |= e[:, 1:]; e = f
    return e

def audit(P, G, K):
    out = {k: [] for k in ("p_rel", "q_rel", "q_rel_band", "q_rel_bulk", "band_err_share", "band_area", "mass_res", "mass_res_label", "outflow_err", "outflow_err_label")}
    for p, g, k in zip(P.astype(np.float64), G.astype(np.float64), K.astype(np.float64)):
        qp, qg = faces(p, k), faces(g, k)
        ex = (qp[0] - qg[0]) ** 2; ey = (qp[1] - qg[1]) ** 2
        B = band(k); bx = B[:, 1:] | B[:, :-1]; by = B[1:] | B[:-1]
        num = ex.sum() + ey.sum(); den = (qg[0] ** 2).sum() + (qg[1] ** 2).sum()
        out["p_rel"].append(np.linalg.norm(p - g) / np.linalg.norm(g)); out["q_rel"].append(np.sqrt(num / den))
        nb = (ex * bx).sum() + (ey * by).sum(); db = (qg[0] ** 2 * bx).sum() + (qg[1] ** 2 * by).sum()
        out["q_rel_band"].append(np.sqrt(nb / db)); out["q_rel_bulk"].append(np.sqrt((num - nb) / (den - db)))
        out["band_err_share"].append(nb / num); out["band_area"].append(B.mean())
        src = h * h                                   # f = 1 integrated over one cell
        out["mass_res"].append(np.sqrt(((div(*qp) - src) ** 2).mean()) / src)
        out["mass_res_label"].append(np.sqrt(((div(*qg) - src) ** 2).mean()) / src)
        tot = lambda b: sum(x.sum() for x in b) * h  # total outflow; exact value = integral of f = 1
        out["outflow_err"].append(abs(tot(qp[2]) - 1)); out["outflow_err_label"].append(abs(tot(qg[2]) - 1))
    return {k: float(np.mean(v)) for k, v in out.items()}

if __name__ == "__main__":
    d = np.load(ROOT / "results/stage3/diagnostics_arrays.npz"); G, K = d["gt"], d["kappa"]
    runs = {"fno_final_stage3": [d["pred"]]}
    for f in sorted(glob.glob(str(ROOT / "results/stage4/runs/*/preds.npz"))):
        runs.setdefault(Path(f).parent.name.rsplit("_s", 1)[0], []).append(np.load(f)["pred"])
    res = {}
    for name, preds in runs.items():
        a = [audit(p, G, K) for p in preds]; res[name] = {k: float(np.mean([x[k] for x in a])) for k in a[0]}; res[name]["n_seeds"] = len(preds)
        r = res[name]; print(f"{name:24s} p {r['p_rel']:.4f}  q {r['q_rel']:.4f} (x{r['q_rel']/r['p_rel']:.1f})  band/bulk {r['q_rel_band']:.3f}/{r['q_rel_bulk']:.3f}"
              f"  band {100*r['band_area']:.0f}% area {100*r['band_err_share']:.0f}% err  mass {r['mass_res']:.2f} (label {r['mass_res_label']:.2f})  outflow {r['outflow_err']:.3f} (label {r['outflow_err_label']:.3f})")
    (ROOT / "results/flux").mkdir(exist_ok=True); json.dump(res, open(ROOT / "results/flux/flux_audit_test.json", "w"), indent=1)
