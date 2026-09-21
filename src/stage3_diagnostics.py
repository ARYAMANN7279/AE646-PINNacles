"""
Physical diagnostics of a trained surrogate on the TEST split (uses a checkpoint saved by stage3_train.py --save-model).

Per test sample (all in physical units, at the checkpoint's training resolution N):
  rel_l2        relative L2 error vs the PDEBench ground truth (with and without D4 test-time symmetrisation)
  pde_res_pred  ||A u_pred - b|| / ||b||   (FV operator A calibrated in solver_vs_fno.py; measures how well the PDE is satisfied)
  pde_res_gt    same for the ground truth (the discretisation floor: the reference itself does not satisfy the FV equations exactly)
  mean_err      relative error of the domain-mean pressure (integral balance: mean(u) ~ total flux balance)
  neg_frac      fraction of pixels with u_pred < 0 (u >= 0 by the maximum principle for f > 0)
  bc_pred/bc_gt mean |u| on the outermost ring relative to the interior mean (Dirichlet u = 0 boundary)
  hetero        std(kappa)/mean(kappa);  area_hi = fraction of the high-permeability phase;  peak = max(u_gt)
  iface_err     mean abs error binned by distance (in cells) to the nearest permeability interface
Also: spectral truncation floor - relative L2 error of the ground truth low-pass filtered to m retained modes
(the best any model with m Fourier modes per spectral layer could do if it only kept those modes).
Writes results/stage3/diagnostics.json and diagnostics_arrays.npz.
Run: python src/stage3_diagnostics.py --ckpt results/stage3/runs/<name>/best_model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

import solvers as S
import stage3_train as T

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--face", default="arithmetic"); ap.add_argument("--bc", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "stage3"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    spec = ck["spec"]; stats = ck["stats"]; res = spec["res"]
    ns = T.parse(["--name", "diag"] + sum([[f"--{k.replace('_','-')}"] + ([str(x) for x in v] if isinstance(v, list) else [str(v)]) for k, v in spec.items()
                                          if k in ("model", "width", "modes", "layers", "hidden", "res")], []))
    model = T.build_model(ns, res).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    data = T.Data(900, dev)
    inp, tgt = data.make(data.k_te, data.u_te, res, stats)
    with torch.no_grad():
        P = model(inp)
    pred = (P[..., 0] * stats["u_std"] + stats["u_mean"]).cpu().numpy()
    gt = (tgt[..., 0] * stats["u_std"] + stats["u_mean"]).cpu().numpy()
    kap = data.k_te[:, ::128 // res, ::128 // res].cpu().numpy()
    # symmetrised prediction
    tta = T.evaluate(model, data, data.k_te, data.u_te, res, stats, tta=True)
    plain = T.evaluate(model, data, data.k_te, data.u_te, res, stats)

    n = len(pred)
    rel = np.linalg.norm((pred - gt).reshape(n, -1), axis=1) / np.linalg.norm(gt.reshape(n, -1), axis=1)
    res_p, res_g = [], []
    for i in range(n):
        res_p.append(S.residual(kap[i], pred[i], 1.0, a.face, a.bc)); res_g.append(S.residual(kap[i], gt[i], 1.0, a.face, a.bc))
    ring = np.zeros((res, res), bool); ring[0], ring[-1], ring[:, 0], ring[:, -1] = True, True, True, True
    def bc_ratio(u): return np.abs(u[:, ring]).mean(1) / np.abs(u[:, ~ring]).mean(1)
    kmin, kmax = kap.min(), kap.max()
    hi = kap > 0.5 * (kmin + kmax)
    dist = np.stack([distance_transform_edt(h) + distance_transform_edt(~h) - 1 for h in hi])   # cells to nearest interface
    bins = [0, 1, 2, 4, 8, 16, 1e9]
    iface = []
    for lo, hi_ in zip(bins[:-1], bins[1:]):
        m = (dist >= lo) & (dist < hi_)
        iface.append(dict(bin=[lo, hi_ if hi_ < 1e8 else None], frac_pixels=float(m.mean()), mean_abs_err_over_peak=float((np.abs(pred - gt)[m] / gt.max(axis=(1, 2), keepdims=True).repeat(res, 1).repeat(res, 2)[m]).mean())))
    # spectral truncation floor of the ground truth
    F = np.fft.rfft2(gt)
    floors = {}
    for m in (2, 4, 6, 8, 12, 16, 24, 32):
        if m > res // 2:
            continue
        G = np.zeros_like(F); G[:, :m, :m] = F[:, :m, :m]; G[:, -m:, :m] = F[:, -m:, :m]
        low = np.fft.irfft2(G, s=(res, res))
        floors[m] = float((np.linalg.norm((low - gt).reshape(n, -1), axis=1) / np.linalg.norm(gt.reshape(n, -1), axis=1)).mean())
    hetero = kap.reshape(n, -1).std(1) / kap.reshape(n, -1).mean(1)
    out = dict(spec=spec, res=res, n=n, rel_l2_mean=float(plain.mean()), rel_l2_tta_mean=float(tta.mean()),
               pde_res_pred_mean=float(np.mean(res_p)), pde_res_gt_mean=float(np.mean(res_g)),
               domain_mean_rel_err=float(np.mean(np.abs(pred.mean((1, 2)) - gt.mean((1, 2))) / gt.mean((1, 2)))),
               neg_frac_pred=float((pred < 0).mean()), neg_frac_gt=float((gt < 0).mean()),
               bc_ratio_pred=float(bc_ratio(pred).mean()), bc_ratio_gt=float(bc_ratio(gt).mean()),
               corr_err_area_hi=float(np.corrcoef(rel, hi.mean((1, 2)))[0, 1]), corr_err_hetero=float(np.corrcoef(rel, hetero)[0, 1]),
               corr_err_peak=float(np.corrcoef(rel, gt.max((1, 2)))[0, 1]), corr_err_pderes=float(np.corrcoef(rel, res_p)[0, 1]),
               interface_error=iface, spectral_floor=floors, kappa_levels=[float(kmin), float(kmax)])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "diagnostics.json", "w"), indent=1)
    np.savez_compressed(Path(a.out) / "diagnostics_arrays.npz", pred=pred, gt=gt, kappa=kap, rel=rel, pde_res_pred=res_p, pde_res_gt=res_g,
                        area_hi=hi.mean((1, 2)), peak=gt.max((1, 2)), hetero=hetero, dist=dist.astype(np.float32))
    print(json.dumps({k: v for k, v in out.items() if k not in ("spec", "interface_error")}, indent=1))


if __name__ == "__main__":
    main()
