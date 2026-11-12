"""
Extra experiment (NOT part of the submitted Stage 3 report): add a soft Dirichlet-boundary penalty to the loss.

The PDE says u = 0 on the boundary, but the PDEBench arrays behave as if that boundary sits `bc` = 1 cell
OUTSIDE the array (see solvers.assemble), so the outermost array pixels are not zero. The penalty therefore
acts on the value linearly extrapolated to the boundary:  u_b = u_0 - bc * (u_1 - u_0)  on all four edges,
where u_0 is the edge pixel and u_1 its inner neighbour.

  loss_i = relL2_i + lam * rms(u_b of prediction) / rms(u_true)        (physical units, per sample)

Implementation: stage3_train.rel_l2_phys is wrapped. The wrapper adds the penalty only while gradients are
enabled (the training loop); evaluation runs under torch.no_grad() and is therefore the plain relative L2.
The submitted training code is not modified.

Run: python src/stage3_bcloss.py --lam 1.0 --name x1 --model fno --layers 4 --sched cosine --loss rel_l2 --aug d4 --tta --steps 5700 --seed 0
     python src/stage3_bcloss.py --check      (boundary penalty of the ground truth and of a checkpoint, no training)
"""
import sys
from pathlib import Path

import numpy as np
import torch

import stage3_train as T

BC = 1.0


def boundary_values(u, bc=BC):
    """u: (B,H,W) physical pressure -> (B, 4H) values extrapolated to the Dirichlet boundary."""
    g = [u[:, 0, :] - bc * (u[:, 1, :] - u[:, 0, :]), u[:, -1, :] - bc * (u[:, -2, :] - u[:, -1, :]),
         u[:, :, 0] - bc * (u[:, :, 1] - u[:, :, 0]), u[:, :, -1] - bc * (u[:, :, -2] - u[:, :, -1])]
    return torch.cat(g, dim=1)


def bc_penalty(u, ref):
    """Per-sample rms(boundary values of u) / rms(ref)."""
    return boundary_values(u).pow(2).mean(1).sqrt() / ref.reshape(ref.shape[0], -1).pow(2).mean(1).sqrt()


def install(lam):
    orig = T.rel_l2_phys

    def wrapped(pred, target, stats):
        out = orig(pred, target, stats)
        if lam > 0 and torch.is_grad_enabled():
            p = (pred[..., 0] * stats["u_std"] + stats["u_mean"])
            t = (target[..., 0] * stats["u_std"] + stats["u_mean"])
            out = out + lam * bc_penalty(p, t)
        return out
    T.rel_l2_phys = wrapped


@torch.no_grad()
def report(name, root, res=64):
    """Boundary penalty of ground truth and of the saved best model on the test split."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = T.Data(900, dev)
    ck = torch.load(Path(root) / name / "best_model.pt", map_location=dev)
    stats = ck["stats"]; spec = ck["spec"]
    ns = T.parse(["--name", "x", "--model", "fno", "--modes", str(spec["modes"]), "--width", str(spec["width"]), "--layers", str(spec["layers"])])
    model = T.build_model(ns, res).to(dev); model.load_state_dict(ck["model_state_dict"]); model.eval()
    inp, tgt = data.make(data.k_te, data.u_te, res, stats)
    P = model(inp)[..., 0] * stats["u_std"] + stats["u_mean"]
    G = tgt[..., 0] * stats["u_std"] + stats["u_mean"]
    return float(bc_penalty(P, G).mean()), float(bc_penalty(G, G).mean())


def check_ground_truth(res=64):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = T.Data(900, dev)
    s = 128 // res
    for nm, u in (("train", data.u_tr), ("val", data.u_va), ("test", data.u_te)):
        G = u[:, ::s, ::s]
        naive = G[:, [0, -1], :].abs().mean().item(), G[:, :, [0, -1]].abs().mean().item()
        print(f"{nm}: extrapolated-boundary penalty of ground truth = {float(bc_penalty(G, G).mean()):.4f}; "
              f"mean |u| on edge pixels = {np.mean(naive):.4f} vs interior mean {float(G[:, 1:-1, 1:-1].mean()):.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--check" in argv:
        check_ground_truth()
        sys.exit()
    lam = 0.0
    if "--lam" in argv:
        i = argv.index("--lam"); lam = float(argv[i + 1]); del argv[i:i + 2]
    install(lam)
    if "--out-root" not in argv:
        argv += ["--out-root", str(T.ROOT / "results" / "extra_bcloss" / "runs")]
    a = T.parse(argv + ["--save-model"])
    r = T.run(a)
    pen_pred, pen_gt = report(a.name, a.out_root, a.res)
    import json
    p = Path(a.out_root) / a.name / "metrics.json"
    m = json.load(open(p)); m.update(bc_lam=lam, bc_penalty_pred_test=pen_pred, bc_penalty_gt_test=pen_gt)
    json.dump(m, open(p, "w"))
    print(f"[{a.name}] lam={lam} bc-penalty pred {pen_pred:.4f} vs ground truth {pen_gt:.4f}", flush=True)
