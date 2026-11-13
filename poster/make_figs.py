"""Re-render the deck and robustness figures as vector PDFs for the A0 poster (writes only into poster/figs).
Run from the repo root: python3 poster/make_figs.py"""
import os, runpy, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "poster" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
_orig = Figure.savefig

def redirected(self, fname, *a, **k):            # every figure goes to poster/figs as PDF (and PNG), nowhere else
    stem = Path(str(fname)).stem
    k.pop("dpi", None)
    _orig(self, OUT / f"{stem}.pdf", *a, **k)
    _orig(self, OUT / f"{stem}.png", *a, dpi=300, **k)
Figure.savefig = redirected

os.chdir(OUT)
runpy.run_path(str(ROOT / "stage3" / "deck_extension" / "make_figures.py"), run_name="__main__")
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "stage3_robust_aggregate.py"), run_name="__main__")
print(sorted(p.name for p in OUT.glob("*.pdf")))

# poster-shaped physics figure (same data as the deck's fig_deck_physics, larger labels)
import numpy as np
import matplotlib.pyplot as plt
D = np.load(ROOT / "results" / "stage3" / "diagnostics_arrays.npz"); rel = D["rel"]; area = D["area_hi"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
ax = axs[0]
ax.hist(rel, bins=np.linspace(0, 0.4, 21), color="#1E8E5A", edgecolor="white")
ax.axvline(np.median(rel), color="#142F4E", ls="--", lw=1.5)
ax.text(0.05, ax.get_ylim()[1] * 0.6, f"typical map\n{np.median(rel):.3f}", fontsize=12, color="#142F4E")
ax.set_xlabel("error of one test map", fontsize=12); ax.set_ylabel("number of maps", fontsize=12)
ax.set_title(f"{(rel < 0.05).mean() * 100:.0f}% of maps under 5% error", fontsize=12.5, color="#142F4E")
ax = axs[1]
ax.scatter(area, rel, s=16, color="#2F7FD1", alpha=.8)
ax.set_xlabel("share of map with high permeability", fontsize=12); ax.set_ylabel("error", fontsize=12)
ax.set_title("worst maps: almost all low permeability", fontsize=12.5, color="#142F4E")
for a_ in axs:
    a_.tick_params(labelsize=11)
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("fig_poster_physics.png")
