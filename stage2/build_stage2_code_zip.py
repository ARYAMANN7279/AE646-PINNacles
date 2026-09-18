"""
Builds the Stage 2 code archive: stage2/PINNacles_Stage2_Code.zip

Contains only the Stage 2 scope (data pipeline, MLP baseline + ablations, initial FNO,
EDA, tests) plus the stored results/figures that back the Stage 2 report, and the
Stage 2 README (stage2/README.md). No model checkpoints, no __pycache__.

Run (from repo root):  python3 stage2/build_stage2_code_zip.py
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
NAME = "PINNacles_Stage2_Code"

SRC = ["download_data.py", "preprocess.py", "models.py", "train.py", "evaluate.py",
       "eda.py", "ablation_mlp.py", "ablation_preprocess.py", "generate_data.py"]
CONFIGS = ["fno.yaml", "mlp.yaml"]
RESULT_FILES = [
    "run_001/test_metrics.json", "run_001/evaluation/eval_metrics.json",
    "run_001/evaluation/error_distribution.png", "run_001/evaluation/sample_predictions.png",
    "run_002/test_metrics.json", "run_002/evaluation/eval_metrics.json",
    "run_002/evaluation/error_distribution.png", "run_002/evaluation/sample_predictions.png",
    "ablation_mlp.json", "ablation_preprocess.json", "eda_metrics.json",
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


with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / NAME
    for f in SRC:
        copy(ROOT / "src" / f, out / "src" / f)
    for f in CONFIGS:
        copy(ROOT / "configs" / f, out / "configs" / f)
    copy(ROOT / "tests" / "__init__.py", out / "tests" / "__init__.py")
    copy(ROOT / "tests" / "test_models.py", out / "tests" / "test_models.py")
    for f in RESULT_FILES:
        copy(ROOT / "results" / f, out / "results" / f)
    for f in (HERE / "figures").iterdir():          # Stage 2 figures only
        if f.suffix in (".pdf", ".png"):
            copy(f, out / "results" / "figures" / f.name)
    for f in ["environment.yml", "pyproject.toml"]:
        copy(ROOT / f, out / f)
    (out / "requirements.txt").write_text(REQUIREMENTS)
    copy(HERE / "README.md", out / "README.md")

    zpath = HERE / f"{NAME}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(NAME) / f.relative_to(out))
    print(f"Wrote {zpath} ({zpath.stat().st_size/1e6:.1f} MB, "
          f"{sum(1 for _ in out.rglob('*') if _.is_file())} files)")
