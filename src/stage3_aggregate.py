"""Collect results/stage3/runs/*/metrics.json into results/stage3/summary.json (one row per run) and print E1 table."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results" / "stage3"


def load():
    rows = {}
    for p in sorted((R / "runs").glob("*/metrics.json")):
        m = json.load(open(p)); s = m["spec"]
        rows[s["name"]] = dict(name=s["name"], model=s["model"], params=m["params"], test=m["test_mean"], val=m["val_mean"],
                               tta=m.get("test_tta_mean"), median=m["test_median"], best_step=m["best_step"], steps_per_s=m["steps_per_s"],
                               train_time_s=m["train_time_s"], at_res=m.get("test_at_res"), spec={k: s[k] for k in
                               ("width", "modes", "layers", "hidden", "res", "n_train", "aug", "steps", "batch", "lr", "wd", "loss", "sched", "seed")})
    return rows


def ms(v):
    v = np.array(v, float); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


if __name__ == "__main__":
    rows = load()
    (R / "summary.json").write_text(json.dumps(rows, indent=1))
    g = defaultdict(list)
    for n, r in rows.items():
        m = re.match(r"e1_(\w+?)_(leg|cos)_(mse|rel_l2)_(none|d4)_s\d", n)
        if m:
            g[m.groups()].append(r)
    print(f"{'model':5} {'sched':4} {'loss':6} {'aug':4} n  test(mean±sd)      tta")
    for k in sorted(g):
        t = ms([r["test"] for r in g[k]]); a = ms([r["tta"] for r in g[k]])
        print(f"{k[0]:5} {k[1]:4} {k[2]:6} {k[3]:4} {len(g[k])}  {t[0]:.4f}±{t[1]:.4f}   {a[0]:.4f}±{a[1]:.4f}")
