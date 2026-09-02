# AE646-PINNacles — Fourier Neural Operator for Parametric Darcy Flow

**AE646: Scientific Machine Learning for Fluid Mechanics — Project theme 8 (Operator learning for parametric PDEs)**

Learn the solution operator of the 2D parametric Darcy flow equation with a Fourier Neural
Operator (FNO), compared against a fully-connected MLP baseline with no spatial inductive
bias, on the real **PDEBench 2D Darcy Flow (β=1.0)** dataset.

> **Team repo.** Reports, code, results, and the presentation all live here so the team can
> work from one place. Stage deliverables (proposal, interim, final report, slides) are in
> [`docs/`](docs/).

## Problem

```
-div(kappa(x,y) * grad(u(x,y))) = f(x,y),   (x,y) in (0,1)^2
u = 0 on the boundary,   f = beta = 1.0
```

`kappa` is a piecewise-constant permeability field (values in {0.1, 1.0}, obtained by
thresholding a smooth Gaussian random field) — this is PDEBench's actual Darcy setup
(Takamoto et al., *PDEBench*, NeurIPS 2022), not the continuous log-permeability field used
in the original Li et al. FNO paper's Darcy dataset. The task is to learn the operator
`G: kappa -> u`.

## Dataset

**Real PDEBench 2D Darcy Flow, β=1.0** — downloaded directly from the official source
(`src/download_data.py`, DaRUS/Uni Stuttgart, DOI 10.18419/darus-2986), verified by checksum
against PDEBench's published manifest. 10,000 genuine samples at native 128×128 resolution.

`src/preprocess.py` draws a reproducible (seed=42), non-overlapping subset — 1000 samples
for train/val (further split 900/100) and 200 held out for test — downsamples to 64×64 by
taking every 2nd grid point for the main train/eval pipeline, and separately keeps the 200
test samples' **native 128×128 fields** aside for a genuine zero-shot super-resolution check.

An optional, clearly-labeled synthetic fallback generator (`src/generate_data.py`) is included
in case the real download is unavailable — see "Optional: synthetic fallback" below. It is
**not** what the results in this repo use.

## Project Structure
```
AE646-PINNacles/
|-- README.md
|-- requirements.txt / environment.yml / pyproject.toml
|-- configs/
|   |-- fno.yaml             # FNO (original): width=64, modes=12, 4 layers
|   |-- mlp.yaml             # MLP baseline: 3x2048 hidden
|   `-- fno_improved.yaml    # FNO (improved): width=128, modes=20, 6 layers
|-- src/
|   |-- download_data.py     # real PDEBench download (checksum-verified)
|   |-- generate_data.py     # optional synthetic fallback (NOT used for reported results)
|   |-- preprocess.py        # subset selection, downsampling, normalization, split
|   |-- models.py            # MLPBaseline, SpectralConv2d, FNO2d
|   |-- train.py             # training loop (physical-unit relative-L2 metric)
|   |-- evaluate.py          # test-set evaluation + visualization
|   |-- superres_eval.py     # zero-shot super-resolution at native 128x128
|   |-- benchmark_speed.py   # real FNO vs MLP vs FDM-solver timing
|   |-- compare_comprehensive.py
|   |-- benchmark_components.py  # per-layer FNO timing with GPU sync
|   `-- generate_figures.py      # all report figures from stored JSON results
|-- tests/                   # pytest suite for models/metrics/data
|-- scripts/                 # md->pdf, pptx build, remote-GPU run helper
|-- results/                 # metrics (JSON) + figures for the 3 runs (checkpoints not tracked)
|-- docs/                    # stage deliverables
|   |-- ae646_handout.pdf    # course project spec
|   |-- PROPOSAL.md / .pdf
|   |-- INTERIM_REPORT.md / .pdf
|   |-- FINAL_REPORT.md / .pdf
|   `-- PRESENTATION.md / .pptx
`-- data/                    # NOT tracked - regenerate with the download + preprocess steps
    |-- raw_pdebench/        # downloaded PDEBench HDF5
    `-- processed/           # train/val/test .npz + test_hires.npz + norm_stats.json
```

## Quick Start

### 1. Setup Environment
```bash
# Option A: Conda
conda env create -f environment.yml
conda activate ae646-fno

# Option B: pip
pip install -r requirements.txt
```

### 2. Download real PDEBench data (~1.25 GB)
```bash
python src/download_data.py
```

### 3. Preprocess
```bash
python src/preprocess.py
```

### 4. Train
```bash
python src/train.py --config configs/fno.yaml
python src/train.py --config configs/mlp.yaml
python src/train.py --config configs/fno_improved.yaml
```
(pass `--wandb` to additionally log to Weights & Biases; off by default so no wandb
account/login is needed to reproduce results)

### 5. Evaluate, super-resolution, speed benchmark
```bash
python3 src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt
python3 src/evaluate.py --config configs/mlp.yaml --checkpoint results/run_002/best_model.pt
python3 src/evaluate.py --config configs/fno_improved.yaml --checkpoint results/run_003_fno_improved/best_model.pt

python3 src/superres_eval.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt
python3 src/superres_eval.py --config configs/fno_improved.yaml --checkpoint results/run_003_fno_improved/best_model.pt
python3 src/benchmark_speed.py
python3 src/benchmark_components.py   # per-layer FNO profiling
python3 src/compare_comprehensive.py
```

### 6. Generate report figures
```bash
python3 src/generate_figures.py       # outputs to results/figures/
```

## Model Details

### FNO (Fourier Neural Operator)
- **Architecture**: Lift (3→width) → N Spectral Conv blocks → Project (width→128→1)
- **Spectral Conv**: 2D FFT → multiply learned weights in Fourier space → IFFT
- **Input**: (B, 64, 64, 3) = [permeability, x_coord, y_coord]; **Output**: (B, 64, 64, 1) = pressure

| Config | width | modes | layers | Params |
|---|---|---|---|---|
| Original (`fno.yaml`) | 64 | 12 | 4 | 4.7M |
| Improved (`fno_improved.yaml`) | 128 | 20 | 6 | 78.8M |

### MLP Baseline
- Flatten (64×64×3) → 3×2048 hidden (GELU) → Flatten output (64×64) — 42.0M parameters
- No spatial inductive bias — treats the field as a dense vector mapping

## Results (real PDEBench test set, 200 samples, relative L2 in physical units)

| Model | Mean Rel L2 | Median | Std | Params |
|-------|------------|--------|-----|--------|
| FNO (original) | 0.0521 | 0.0398 | 0.0451 | 4.7M |
| MLP baseline | 0.0820 | 0.0695 | 0.0456 | 42.0M |
| FNO (improved) | 0.0456 | 0.0307 | 0.0487 | 78.8M |

Zero-shot at native 128×128 (real PDEBench ground truth, no retraining): FNO (original)
0.0594, FNO (improved) 0.0563 — the MLP cannot be evaluated this way (fixed input size).

Measured inference speed (`results/benchmark_speed.json`): FNO 0.71 ms/sample and MLP
0.13 ms/sample on GPU vs 1030 ms/sample for a scipy sparse FDM solve on CPU. See
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) for full discussion, including why MLP is
actually *faster* per-sample than FNO here despite having 9× more parameters.

## Reproducibility
- Seed: 42 everywhere (data subset selection, train/val split, model init)
- Normalization stats computed from the training split only, saved to `norm_stats.json`
- Relative-L2 metric is computed in physical (denormalized) units, matching the
  literature-standard convention — see the docstring of `physical_rel_l2` in `src/train.py`
  for why this matters
- Training was run on an NVIDIA RTX PRO 6000 (CUDA); results were cross-checked against an
  independent run on Apple Silicon (MPS) and matched closely (FNO 0.0521 vs 0.0544, MLP
  0.0820 vs 0.0840 mean rel L2), confirming the pipeline is deterministic/reproducible
  across hardware
- Run the tests with `pytest` from the repo root

## Documents (in `docs/`)
Reports are written in LaTeX (`.tex`). Compile with tectonic (recommended) or any TeX distribution:

```bash
# Install tectonic once
brew install tectonic

# Compile reports
cd docs
tectonic INTERIM_REPORT.tex          # -> INTERIM_REPORT.pdf
tectonic FINAL_REPORT.tex            # -> FINAL_REPORT.pdf
tectonic CONTRIBUTION_STATEMENT.tex  # -> CONTRIBUTION_STATEMENT.pdf
```

Rebuild the slide deck:
```bash
cd docs
python3 build_presentation.py        # -> docs/PINNacles_Stage1_Presentation.pptx
```

## Optional: synthetic fallback
`src/generate_data.py` self-generates a Darcy dataset (piecewise-constant permeability,
finite-difference solve) matching PDEBench's conventions, for use only if the real download
is unavailable. It is explicitly permitted by the course handout but is **not** the source
of any number reported in this repo.

## References
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICML 2021
- Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022
- PDEBench data: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- NeuralOperator: https://github.com/neuraloperator/neuraloperator
