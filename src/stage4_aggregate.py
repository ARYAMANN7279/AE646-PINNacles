"""
Aggregate results/stage4/runs/*/metrics.json -> results/stage4/summary.json and a printed table.

Per configuration (run name without the _s<seed> suffix): mean and sd over seeds of the test error, the error on the
hard maps and on the rest, the error on the 32 / 64 / 128 grids and, when present, on each out-of-distribution set.
Ensemble uncertainty (configurations with preds.npz for every seed): error of the seed-ensemble mean, and how well the
ensemble spread ranks the per-map error (Spearman rank correlation; overlap of the 10 worst maps with the 10 most
uncertain ones). Test truth: results/stage3/diagnostics_arrays.npz (gt, 64 x 64, test order).
Run from the repo root: python src/stage4_aggregate.py
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R4 = ROOT / "results" / "stage4"


def ms(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    runs = defaultdict(list)
    for p in sorted(glob.glob(str(R4 / "runs" / "*" / "metrics.json"))):
        name = Path(p).parent.name
        runs[re.sub(r"_s\d+$", "", name)].append((name, json.load(open(p))))
    S = {}
    for cfg, rs in sorted(runs.items()):
        m = [r for _, r in rs]
        d = {"arch": m[0]["arch"], "params": m[0]["params"], "test": ms([r["test_mean"] for r in m]),
             "hard": ms([r["test_hard"] for r in m]), "rest": ms([r["test_rest"] for r in m]),
             "train_time_s": ms([r["train_time_s"] for r in m])}
        if "test_at_res" in m[0]:
            d["at_res"] = {k: ms([r["test_at_res"][k] for r in m]) for k in ("32", "64", "128")}
        if "ood" in m[0]:
            d["ood"] = {k: ms([r["ood"][k]["mean"] for r in m]) for k in m[0]["ood"]}
        S[cfg] = d
    # ensemble uncertainty
    diag = ROOT / "results" / "stage3" / "diagnostics_arrays.npz"      # the 200 test truths at 64 x 64, in test order
    if diag.exists():
        truth = np.load(diag)["gt"].astype(np.float64); te = range(len(truth))
        tn = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
        for cfg, rs in runs.items():
            files = [R4 / "runs" / n / "preds.npz" for n, _ in rs]
            if len(files) < 2 or not all(f.exists() for f in files):
                continue
            P = np.stack([np.load(f)["pred"].astype(np.float64) for f in files])          # (seeds, 200, 64, 64)
            mean = P.mean(0)
            err_ens = np.linalg.norm((mean - truth).reshape(len(te), -1), axis=1) / tn
            err_single = np.array([np.linalg.norm((p - truth).reshape(len(te), -1), axis=1) / tn for p in P])
            spread = np.linalg.norm(P.std(0).reshape(len(te), -1), axis=1) / tn
            worst = set(np.argsort(-err_ens)[:10]); unsure = set(np.argsort(-spread)[:10])
            S[cfg]["ensemble"] = {"members": len(files), "ens_error": float(err_ens.mean()), "single_error_mean": float(err_single.mean()),
                                  "spearman_spread_vs_error": spearman(spread, err_ens), "top10_overlap": len(worst & unsure)}
    R4.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(R4 / "summary.json", "w"), indent=1)
    print(f"{'config':22s} {'params':>7s} {'test':>15s} {'hard':>13s} {'rest':>8s}  res32/64/128")
    for cfg, d in S.items():
        r = d.get("at_res")
        print(f"{cfg:22s} {d['params']/1e6:6.2f}M {d['test']['mean']:.4f}±{d['test']['sd']:.4f} {d['hard']['mean']:.3f}±{d['hard']['sd']:.3f} "
              f"{d['rest']['mean']:.4f}  " + (f"{r['32']['mean']:.3f}/{r['64']['mean']:.3f}/{r['128']['mean']:.3f}" if r else "-")
              + (f"  ens {d['ensemble']['ens_error']:.4f} vs {d['ensemble']['single_error_mean']:.4f}, rho {d['ensemble']['spearman_spread_vs_error']:.2f}, top10 {d['ensemble']['top10_overlap']}" if "ensemble" in d else "")
              + ("  OOD " + " ".join(f"{k}:{v['mean']:.3f}" for k, v in d["ood"].items()) if "ood" in d else ""))


if __name__ == "__main__":
    main()
