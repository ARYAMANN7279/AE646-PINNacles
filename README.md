# AE646-PINNacles — Fourier Neural Operator for Parametric Darcy Flow

**AE646: Scientific Machine Learning for Fluid Mechanics — Project theme 8 (Operator learning for parametric PDEs)**

Learn the solution operator of the 2D parametric Darcy flow equation with a Fourier Neural
Operator (FNO), compared against a fully-connected MLP baseline with no spatial inductive
bias, on the real **PDEBench 2D Darcy Flow (β=1.0)** dataset.

> **Team repo.** Reports, code, results, and the presentation all live here so the team can
> work from one place. Stage deliverables (proposal, interim, final report, slides) are in
> [`stage1/`](stage1/), [`stage2/`](stage2/), [`stage3/`](stage3/).

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
|   |-- eda.py                   # dataset EDA + kappa-heterogeneity vs error correlation
|   |-- ablation_mlp.py          # real MLP depth/optimizer ablation (retrains from scratch)
|   |-- ablation_preprocess.py   # real MLP preprocessing ablation (3 seeds/variant)
|   `-- generate_figures.py      # all report figures from stored JSON results
|-- tests/                   # pytest suite for models/metrics/data
|-- scripts/                 # remote-GPU run helper
|-- results/                 # metrics (JSON) + figures for the 3 runs, ablation, EDA (checkpoints not tracked)
|-- ae646_handout.pdf        # course project spec
|-- stage1/                  # Stage 1: proposal (LaTeX + PDF) and proposal deck (pptx/pdf + builder)
|-- stage2/                  # Stage 2: interim report (.tex/.pdf), 10-slide deck, Stage 2 README, code ZIP + single notebook
|-- stage3/                  # Stage 3: final report, contribution statement/AI declaration, Beamer deck
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
python3 src/eda.py                    # dataset EDA + error-vs-heterogeneity correlation
python3 src/ablation_mlp.py --config configs/mlp.yaml   # real depth/optimizer ablation
python3 src/ablation_preprocess.py                      # real preprocessing ablation (needs raw HDF5)
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
[`stage3/PINNacles_Stage3_FinalReport.pdf`](stage3/PINNacles_Stage3_FinalReport.pdf) for full discussion, including why MLP is
actually *faster* per-sample than FNO here despite having 9× more parameters.

**Dataset EDA** (`results/eda_metrics.json`, `src/eda.py`): κ is exactly bimodal at
{0.1, 1.0}; high-permeability area fraction varies widely across samples (mean 0.48 ±
0.21); 180/200 native-resolution test samples have exactly one connected high-κ region
(multi-region samples are rare). Correlating FNO's per-sample test error against κ
heterogeneity gives only a weak relationship (region count r=0.24, interface perimeter
r=−0.17, wrong sign) — the single worst test sample instead has *zero* connected
regions (a near-degenerate, almost-uniform field). See
[`stage3/PINNacles_Stage3_FinalReport.pdf`](stage3/PINNacles_Stage3_FinalReport.pdf) §3.4/§9.2 for the full, corrected
discussion (this revises an earlier, untested "many regions → high error" claim).

**MLP baseline ablation** (`results/ablation_mlp.json`, `src/ablation_mlp.py`) — real
re-trained runs, not asserted: 1-layer 0.1008, 2-layer 0.0796, 3-layer/AdamW (reported
baseline) 0.0857, 3-layer/SGD+momentum 0.1209. SGD is clearly worse (justifies AdamW);
depth beyond 2 layers shows diminishing/non-monotonic returns.

**MLP preprocessing ablation** (`results/ablation_preprocess.json`, 3 seeds per variant,
identical split/target): reference 0.0842 ± 0.0002, no coordinate channels 0.0843 ± 0.0011,
no input normalisation 0.0864 ± 0.0018, no target normalisation 0.0917 ± 0.0012 —
normalising the target matters (+9% error without it); coordinates don't matter for the MLP.

**Training-loss / schedule note:** `train.py` optimises MSE on standardised pressure (the
relative-L2 error is the *reported metric*, computed in physical units), and steps
`CosineAnnealingLR(T_max=epochs)` once per mini-batch, so the LR cycles between 1e-3 and 0
every 2·T_max steps (~28 cycles per 100-epoch run) rather than decaying once. Kept as-is so
the committed results reproduce; a per-epoch schedule is listed as future work.

## Computational Environment
The pipeline is a set of Python scripts (`src/`), run from the command line. The Stage 2 archive
additionally ships `stage2/PINNacles_Stage2_Notebook.ipynb`, one notebook that combines the Stage 2
scripts (data -> train -> evaluate -> EDA -> ablations -> tests); it was executed end to end on the GPU
workstation below (0 errors; fresh seeded run: FNO 0.0528, MLP 0.0840 mean rel L2). No Colab/Kaggle
service was used. Two machines were involved:
- **Local development**: macOS, Apple Silicon (MPS backend), Python 3.11/3.14, PyTorch 2.3+.
  Used for coding, the pytest suite, and cross-checking numbers.
- **Training/benchmarking**: a remote Linux workstation (4×NVIDIA RTX PRO 6000, CUDA 12.x),
  accessed via SSH, running the identical `src/` scripts against a shared conda environment
  (Python 3.11, PyTorch 2.3, CUDA-enabled). This machine produced the FNO/MLP training runs,
  the GPU inference-speed benchmark, and (for the layer-profiling numbers specifically) an
  MPS run on the local Mac — both are noted explicitly wherever they're reported.
- No GPU/cloud credits or paid services were used.

## Reproducibility
- Seed: 42 everywhere (data subset selection, train/val split, model init, batch order).
  Model-init/batch-order seeding was added to `train.py` after the officially-reported
  `run_001`/`run_002`/`run_003_fno_improved` checkpoints were trained; re-running training
  now reproduces those numbers closely but not bit-for-bit (see the MLP ablation's own
  reproducibility note in `stage3/PINNacles_Stage3_FinalReport.pdf` §6.3 for a measured example of the gap).
- Normalization stats computed from the training split only, saved to `norm_stats.json`
- Relative-L2 metric is computed in physical (denormalized) units, matching the
  literature-standard convention — see the docstring of `physical_rel_l2` in `src/train.py`
  for why this matters
- Training was run on an NVIDIA RTX PRO 6000 (CUDA); results were cross-checked against an
  independent run on Apple Silicon (MPS) and matched closely (FNO 0.0521 vs 0.0544, MLP
  0.0820 vs 0.0840 mean rel L2); a later seeded CUDA re-run (the Stage 2 notebook) gave FNO 0.0528
  and MLP 0.0840. Differences of a few 1e-3 in mean error are the run-to-run / cross-hardware noise floor
- Run the tests with `pytest` from the repo root

## Documents (in `stage1/`, `stage2/`, `stage3/`)
Each stage folder holds the submitted files. Every report/deck is written in LaTeX and the
`.tex` has the same basename as its PDF, so compiling it regenerates the submitted file.
Each stage folder is self-contained: the figures its LaTeX uses are copied into
`stage2/figures/` and `stage3/figures/` (copies of `results/figures/*.pdf`; edit the LaTeX
freely without touching the results). If you re-run `src/generate_figures.py`, copy the
updated PDFs over.
Compile with tectonic (recommended) or any TeX distribution:

```bash
brew install tectonic                                   # once
cd stage2 && tectonic PINNacles_Stage2_InterimReport.tex
cd stage3 && tectonic PINNacles_Stage3_FinalReport.tex
cd stage3 && tectonic PINNacles_Stage3_ContributionStatement.tex
cd stage3 && tectonic PINNacles_Stage3_FinalPresentation.tex   # Beamer deck
```

The Stage 1 and Stage 2 decks are `.pptx` files built with `python-pptx`
(`pip install python-pptx`):
```bash
python3 stage1/build_stage1_presentation.py
python3 stage2/build_stage2_presentation.py
```
`stage2/` also holds the Stage 2 submission code archive `PINNacles_Stage2_Code.zip` and its
Stage 2 `README.md` (both scoped to Stage 2; rebuild with `python3 stage2/build_stage2_code_zip.py`).
This repo root is the live, full code.

## Optional: synthetic fallback
`src/generate_data.py` self-generates a Darcy dataset (piecewise-constant permeability,
finite-difference solve) matching PDEBench's conventions, for use only if the real download
is unavailable. It is explicitly permitted by the course handout but is **not** the source
of any number reported in this repo.

## References
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021
- Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022
- PDEBench data: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- NeuralOperator: https://github.com/neuraloperator/neuraloperator
