"""
Aggregate the post-presentation studies (results/stage3_extra/runs/*/metrics.json written by stage3_robust.py)
and the matching clean-data baselines (results/stage3/runs/e1_*_cos_rel_l2_d4_s*, full recipe, 5,700 steps)
into results/stage3_extra/summary.json and two figures.

  noise study    : test error on CLEAN maps of models trained on noisy pressure labels (noise RMS = s x map RMS)
  adaptive study : self-adjusting per-sample weights vs the plain loss, overall and on the hard (low-permeability) maps
Hard maps = test maps whose high-permeability area share is below 5 % (8 of the 200 test maps).
Run from the repo root: python src/stage3_robust_aggregate.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "results" / "stage3_extra"
BASE = ROOT / "results" / "stage3" / "runs"
NAVY, BLUE, GREEN, ORANGE, GREY = "#142F4E", "#2F7FD1", "#1E8E5A", "#C96F1E", "#5F6B7A"


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=0)), len(v)


def main():
    S = {}
    base = {"fno": load(str(BASE / "e1_fno_cos_rel_l2_d4_s*/metrics.json")),
            "mlp": load(str(BASE / "e1_mlp_cos_rel_l2_d4_s*/metrics.json"))}
    # ---- noise study
    noise = {}
    for m in ("fno", "mlp"):
        noise[m] = {"0": ms([r["test_mean"] for r in base[m]])}
        for lvl in ("0.01", "0.05", "0.1"):
            runs = load(str(EX / "runs" / f"rb_{m}_n{lvl}_s*/metrics.json"))
            if runs:
                noise[m][lvl] = ms([r["test_mean"] for r in runs])
    S["noise"] = {m: {k: {"mean": v[0], "sd": v[1], "n": v[2]} for k, v in d.items()} for m, d in noise.items()}
    # ---- adaptive weights
    ad = {}
    ref = base["fno"]
    anyrb = load(str(EX / "runs/rb_*/metrics.json"))
    area = np.array(anyrb[0]["test_area_hi"]) if anyrb else np.array([])
    for tag, runs in (("plain", None), ("alpha1", load(str(EX / "runs/rb_fno_ad1_s*/metrics.json"))),
                      ("alpha2", load(str(EX / "runs/rb_fno_ad2_s*/metrics.json")))):
        if tag == "plain":
            runs = ref                                   # clean-data full-recipe runs (3 seeds), same protocol
        if not runs:
            continue
        if area.size == 0 and runs[0].get("test_area_hi"):
            area = np.array(runs[0]["test_area_hi"])
        per = np.array([r["test_per_sample"] for r in runs])            # (seeds, 200)
        d = {"n": len(runs), "all": ms(per.mean(1)), "median": ms(np.median(per, 1))}
        if area.size:
            hard = area < 0.05
            d["hard"] = ms(per[:, hard].mean(1)); d["rest"] = ms(per[:, ~hard].mean(1)); d["n_hard"] = int(hard.sum())
            worst = np.argsort(-per.mean(0))[:10]
            d["worst10"] = ms(per[:, worst].mean(1))
        ad[tag] = {k: ({"mean": v[0], "sd": v[1], "n": v[2]} if isinstance(v, tuple) else v) for k, v in d.items()}
    S["adaptive"] = ad
    tj = EX / "train_hard_maps.json"                                   # indices of the 27 training maps with < 5 % high-permeability area
    if tj.exists():
        hard_idx = np.array(json.load(open(tj))["hard"])
        mask = np.zeros(900, bool); mask[hard_idx] = True
        S["weights"] = {}
        for tag, pat in (("alpha1", "rb_fno_ad1_s*"), ("alpha2", "rb_fno_ad2_s*")):
            W = np.array([r["final_weights"] for r in load(str(EX / "runs" / pat / "metrics.json"))])
            S["weights"][tag] = {"n_hard_train": int(mask.sum()), "hard": float(W[:, mask].mean()), "other": float(W[:, ~mask].mean())}
    (EX).mkdir(parents=True, exist_ok=True)
    json.dump(S, open(EX / "summary.json", "w"), indent=1)

    # ---- figure 1: noise
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    xs = [0, 1, 5, 10]
    for m, c, lab in (("fno", GREEN, "FNO"), ("mlp", ORANGE, "MLP")):
        pts = [(x, S["noise"][m][k]) for x, k in zip(xs, ("0", "0.01", "0.05", "0.1")) if k in S["noise"][m]]
        ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=[p[1]["sd"] for p in pts], marker="o", color=c,
                    lw=2, ms=7, capsize=3, label=lab)
    ax.plot(xs, [x / 100 for x in xs], "--", color=GREY, label="error of the noisy labels themselves")
    ax.set_xlabel("Noise added to training pressure labels (% of each map's size)", fontsize=11)
    ax.set_ylabel("Test error on clean maps", fontsize=11)
    ax.set_xticks(xs); ax.grid(alpha=.25); ax.legend(fontsize=10, frameon=False); ax.tick_params(labelsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(EX / "fig_noise.png"); plt.close(fig)

    # ---- figure 2: adaptive weights
    def make_adaptive(figsize, fname):
        if "alpha1" in ad and "hard" in ad.get("plain", {}):
            fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=200)
            tags = [t for t in ("plain", "alpha1", "alpha2") if t in ad]
            names = {"plain": "plain loss", "alpha1": "adaptive (a = 1)", "alpha2": "adaptive (a = 2)"}
            cols = {"plain": GREY, "alpha1": GREEN, "alpha2": BLUE}
            groups = [("all", "all 200 maps"), ("rest", "other maps"), ("hard", "hard maps\n(< 5 % high perm.)")]
            w = 0.8 / len(tags)
            for i, t in enumerate(tags):
                axs[0].bar(np.arange(len(groups)) + i * w, [ad[t][g]["mean"] for g, _ in groups], w, yerr=[ad[t][g]["sd"] for g, _ in groups],
                           color=cols[t], label=names[t], capsize=2)
            axs[0].set_xticks(np.arange(len(groups)) + w * (len(tags) - 1) / 2); axs[0].set_xticklabels([n for _, n in groups], fontsize=10)
            axs[0].set_ylabel("Test error", fontsize=11); axs[0].legend(fontsize=9, frameon=False); axs[0].set_yscale("log")
            axs[0].set_title("Does weighting the hard maps help them?", fontsize=11, color=NAVY)
            if "weights" in S:
                xs_ = np.arange(2); wd = 0.35
                for i, t in enumerate(("alpha1", "alpha2")):
                    axs[1].bar(xs_ + i * wd, [S["weights"][t]["hard"], S["weights"][t]["other"]], wd, color=cols[t], label=names[t])
                axs[1].axhline(1.0, color=GREY, ls="--", lw=1)
                axs[1].set_xticks(xs_ + wd / 2); axs[1].set_xticklabels([f"hard training maps\n({S['weights']['alpha1']['n_hard_train']} of 900)", "all other\ntraining maps"], fontsize=10)
                axs[1].set_ylabel("final loss weight (1 = equal)", fontsize=11); axs[1].legend(fontsize=9, frameon=False)
                axs[1].set_title("The scheme finds the hard maps by itself", fontsize=11, color=NAVY)
            for a_ in axs:
                for s_ in ("top", "right"):
                    a_.spines[s_].set_visible(False)
                a_.tick_params(labelsize=10)
            fig.tight_layout(); fig.savefig(EX / fname); plt.close(fig)
    make_adaptive((10.5, 3.8), "fig_adaptive.png")
    make_adaptive((12.1, 2.85), "fig_adaptive_slide.png")
    print(json.dumps(S, indent=1)[:3000])


if __name__ == "__main__":
    main()
