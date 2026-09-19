"""
Builds the Stage 2 code submission from an explicit file list:

  stage2/PINNacles_Stage2_Code.zip
      PINNacles_Stage2_Code/
          PINNacles_Stage2_Notebook.ipynb   one notebook that runs the whole pipeline
          README.md, requirements.txt, pyproject.toml
          src/  configs/  tests/  results/  (scripts, hyper-parameters, tests, stored results)
  stage2/PINNacles_Stage2_Notebook.ipynb    (same notebook, for convenience)

The notebook embeds the exact staged source files (%%writefile cells), so the notebook and the
scripts cannot drift apart. Only Stage 2 scope is included; no checkpoints, no __pycache__.

Run (from repo root):  python3 stage2/build_stage2_code_zip.py
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
NAME = "PINNacles_Stage2_Code"
NB_NAME = "PINNacles_Stage2_Notebook.ipynb"

SRC = ["download_data.py", "preprocess.py", "models.py", "train.py", "evaluate.py",
       "ablation_mlp.py", "ablation_preprocess.py", "generate_data.py"]
CONFIGS = ["fno.yaml", "mlp.yaml"]
RESULT_FILES = [
    "run_001/test_metrics.json", "run_001/evaluation/eval_metrics.json",
    "run_001/evaluation/error_distribution.png", "run_001/evaluation/sample_predictions.png",
    "run_002/test_metrics.json", "run_002/evaluation/eval_metrics.json",
    "run_002/evaluation/error_distribution.png", "run_002/evaluation/sample_predictions.png",
    "ablation_mlp.json", "ablation_preprocess.json",
]
REQUIREMENTS = """numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4
torch==2.3.0
h5py==3.11.0
tqdm==4.66.4
pyyaml==6.0.1
wandb==0.17.3
requests==2.32.3
pytest
"""


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def replace_once(text, old, new):
    assert old in text, f"expected text not found: {old[:60]!r}"
    return text.replace(old, new, 1)


def stage2_preprocess(text):
    """Neutral wording for the native-resolution test fields (Stage 2 scope)."""
    text = replace_once(
        text,
        "so a trained 64x64 model's\n   zero-shot super-resolution behaviour can be checked against real\n"
        "   PDEBench ground truth (not synthetic/fabricated data).",
        "so the results can also be examined at the\n   original resolution against real PDEBench ground truth.")
    return replace_once(
        text,
        "# Save NATIVE 128x128 test fields (raw physical units, un-normalized) for the\n"
        "    # zero-shot super-resolution check - real PDEBench ground truth, not fabricated.",
        "# Save the NATIVE 128x128 test fields (raw physical units, un-normalized) for\n"
        "    # analysis at the original resolution - real PDEBench ground truth.")


# ── notebook ──────────────────────────────────────────────────────────────────
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writefile_cell(rel, content):
    return code(f"%%writefile {rel}\n{content}")


HELPER = '''
import json, os, subprocess, sys
from pathlib import Path
from IPython.display import Image, display

for d in ("src", "configs", "tests", "results/figures"):      # folders the notebook writes into
    os.makedirs(d, exist_ok=True)

def sh(cmd, tail=None):
    """Run a shell command, hide progress-bar noise, print (the tail of) its output, fail loudly on error."""
    if cmd.startswith("python "):                      # use the notebook's own interpreter
        cmd = f'"{sys.executable}" ' + cmd[len("python "):]
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).replace("\\r", "\\n").splitlines()
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith(("Training:", "Evaluating:", "md5 "))]
    print("\\n".join(lines[-tail:] if tail else lines))
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
'''

SHOW_RESULTS = '''
def show_table(run, label):
    e = json.load(open(f"results/{run}/evaluation/eval_metrics.json"))
    print(f"{label:22s} mean {e['mean_rel_l2']:.4f}  median {e['median_rel_l2']:.4f}  "
          f"std {e['std_rel_l2']:.4f}  min {e['min_rel_l2']:.4f}  max {e['max_rel_l2']:.4f}")

print("Relative L2 error, 200 held-out test samples (physical units)")
show_table("run_001", "FNO")
show_table("run_002", "MLP baseline")
display(Image("results/run_001/evaluation/sample_predictions.png", width=650))
display(Image("results/run_001/evaluation/error_distribution.png", width=550))
'''

SHOW_ABL = '''
for f, title in [("results/ablation_mlp.json", "MLP depth / optimiser ablation"),
                 ("results/ablation_preprocess.json", "MLP preprocessing ablation")]:
    print(title)
    for r in json.load(open(f)):
        val = r.get("test_rel_l2_mean", r.get("test_mean_rel_l2_avg"))
        print(f"  {r['name']:38s} {val:.4f}")
for fig in ["results/figures/fig7_mlp_ablation.png", "results/figures/fig10_preprocess_ablation.png"]:
    display(Image(fig, width=500))
'''


def build_notebook(stage):
    cells = [
        md("""
# AE646 Stage 2 - FNO vs MLP for Parametric Darcy Flow (Team PINNacles)

One notebook that runs the whole Stage 2 pipeline end to end: **data download and preprocessing -> MLP baseline
and FNO training -> evaluation -> dataset analysis -> ablations -> unit tests**.

* **Run all cells top to bottom.** A GPU is needed for training in reasonable time (about 1-2 GB of downloads;
  the full run takes tens of minutes on a GPU and is impractically slow on CPU).
* The cells in section 1 write the project source files (`src/`, `configs/`, `tests/`) to the working directory,
  so this notebook is self-contained. The same files are also provided as ordinary scripts in the code archive.
* Re-running overwrites `results/` with freshly computed numbers; they match the stored report values closely
  but not bit-for-bit (see the note at the end).
"""),
        code("%pip install -q numpy scipy matplotlib h5py tqdm pyyaml wandb requests pytest"),
        code(HELPER),
        md("""
## 1. Project files
Configuration files, source code and tests are written to disk by the next cells.
"""),
    ]
    for c in CONFIGS:
        cells.append(writefile_cell(f"configs/{c}", (stage / "configs" / c).read_text()))
    for s in SRC + ["eda.py"]:
        cells.append(writefile_cell(f"src/{s}", (stage / "src" / s).read_text()))
    cells.append(writefile_cell("pyproject.toml", (stage / "pyproject.toml").read_text()))
    cells.append(writefile_cell("tests/__init__.py", (stage / "tests" / "__init__.py").read_text() or "\n"))
    cells.append(writefile_cell("tests/test_models.py", (stage / "tests" / "test_models.py").read_text()))
    cells += [
        md("""
## 2. Data
Downloads the real PDEBench 2D Darcy flow file (checksum-verified), then builds the reproducible 900 / 100 / 200
train / validation / test split at 64x64 with standardisation and coordinate channels.
"""),
        code('sh("python src/download_data.py", tail=6)'),
        code('sh("python src/preprocess.py", tail=12)'),
        md("""
## 3. Training
FNO (4.7 M parameters) and the MLP baseline (42 M parameters), 100 epochs each; the checkpoint with the lowest
validation error is kept and evaluated on the 200 test samples.
"""),
        code('os.makedirs("results/run_001", exist_ok=True); os.makedirs("results/run_002", exist_ok=True)\n'
             'sh("python src/train.py --config configs/fno.yaml", tail=6)'),
        code('sh("python src/train.py --config configs/mlp.yaml", tail=6)'),
        md("## 4. Evaluation"),
        code('sh("python src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt", tail=8)\n'
             'sh("python src/evaluate.py --config configs/mlp.yaml --checkpoint results/run_002/best_model.pt", tail=8)'),
        code(SHOW_RESULTS),
        md("## 5. Dataset analysis"),
        code('sh("python src/eda.py", tail=14)\ndisplay(Image("results/figures/fig8_eda_dataset.png", width=650))'),
        md("""
## 6. Ablations (MLP baseline)
Depth / optimiser variants and preprocessing variants (3 seeds each), all retrained from scratch on the same split.
Set `RUN_ABLATIONS = False` to skip this section (it is the slowest part).
"""),
        code('RUN_ABLATIONS = True\n'
             'if RUN_ABLATIONS:\n'
             '    sh("python src/ablation_mlp.py --config configs/mlp.yaml", tail=8)\n'
             '    sh("python src/ablation_preprocess.py", tail=8)\n'
             + "\n".join("    " + l for l in SHOW_ABL.strip("\n").splitlines())),
        md("## 7. Unit tests"),
        code('sh("python -m pytest -q", tail=6)'),
        md("""
## Notes
* The random seed is 42 (data split, model initialisation, batch order). The two training runs behind the numbers in
  the report were made before model-initialisation seeding was added, so a fresh run gives results close to, but not
  exactly equal to, the stored ones (differences of about 0.004 in mean error were observed).
* Training loss is mean-squared error on standardised pressure; the reported relative L2 error is computed after
  converting back to physical units.
"""),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


# ── package ───────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / NAME
    for f in SRC:
        copy(ROOT / "src" / f, out / "src" / f)
    pre = out / "src" / "preprocess.py"
    pre.write_text(stage2_preprocess(pre.read_text()))
    copy(HERE / "eda_stage2.py", out / "src" / "eda.py")                     # dataset-only EDA
    for f in CONFIGS:
        copy(ROOT / "configs" / f, out / "configs" / f)
    copy(ROOT / "tests" / "__init__.py", out / "tests" / "__init__.py")
    copy(ROOT / "tests" / "test_models.py", out / "tests" / "test_models.py")
    for f in RESULT_FILES:
        copy(ROOT / "results" / f, out / "results" / f)
    copy(HERE / "data" / "eda_metrics.json", out / "results" / "eda_metrics.json")
    for f in (HERE / "figures").iterdir():                                    # Stage 2 figures only
        if f.suffix == ".png":                                                # PNG only (PDFs are for LaTeX)
            copy(f, out / "results" / "figures" / f.name)
    copy(ROOT / "pyproject.toml", out / "pyproject.toml")
    (out / "requirements.txt").write_text(REQUIREMENTS)
    copy(HERE / "README.md", out / "README.md")

    nb = build_notebook(out)
    (out / NB_NAME).write_text(json.dumps(nb, indent=1))
    (HERE / NB_NAME).write_text(json.dumps(nb, indent=1))

    zpath = HERE / f"{NAME}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(NAME) / f.relative_to(out))
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"Wrote {zpath} ({zpath.stat().st_size/1e6:.1f} MB, {n} files) and {HERE / NB_NAME}")
