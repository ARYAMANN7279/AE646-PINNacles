# AE646 Stage 2 - Fourier Neural Operator for Parametric Darcy Flow (Team PINNacles)

**AE646: Scientific Machine Learning for Fluid Mechanics - Project theme 8 (operator learning for parametric PDEs).**
This is the Stage 2 (interim) code submission: real data pipeline, MLP baseline (with ablations),
and an initial FNO with preliminary results.

## Problem
```
-div(kappa(x,y) * grad(u(x,y))) = f(x,y),   (x,y) in (0,1)^2,   u = 0 on the boundary,   f = 1
```
`kappa` is a piecewise-constant permeability field (values in {0.1, 1.0}); the task is to learn the
operator `G: kappa -> u`. Baseline: fully-connected MLP. SciML model: Fourier Neural Operator (FNO).

## Computational environment / platform
- **No notebooks** (no Colab / Kaggle / Jupyter): everything runs as plain Python scripts from the command line.
- **Local development and unit tests:** macOS, Apple Silicon (MPS backend), Python 3.11+, PyTorch >= 2.3.
- **Training and ablations:** a remote Linux workstation with an NVIDIA RTX PRO 6000 GPU (CUDA), accessed over SSH,
  running the identical scripts (Python 3.11, PyTorch 2.x + CUDA).
- No paid cloud services were used. The scripts use whichever accelerator is available (MPS on the Mac,
  CUDA on the workstation, otherwise CPU; training is practical on a GPU but slow on CPU).

## Dataset
Real **PDEBench 2D Darcy Flow (beta = 1.0)**, DOI 10.18419/darus-2986: 10,000 samples at 128x128
(~1.3 GB HDF5). `src/download_data.py` downloads it with checksum verification. `src/preprocess.py`
draws a reproducible (seed 42) non-overlapping subset - 1000 train/val samples (split 900 / 100) and
200 held-out test samples - subsamples every 2nd grid point to 64x64, standardises with training-set
statistics only, and adds (x, y) coordinate channels (input = [kappa, x, y], shape 64x64x3).
The 200 test samples are also saved at native 128x128 (`test_hires.npz`).

## Project structure
```
configs/            fno.yaml, mlp.yaml                      (model + training hyper-parameters)
src/
  download_data.py  checksum-verified PDEBench download
  preprocess.py     subset / split / downsample / normalise -> data/processed/*.npz
  models.py         MLPBaseline, SpectralConv2d, FNO2d
  train.py          training loop (best-validation checkpoint, seeded)
  evaluate.py       test metrics + best/median/worst sample plots
  eda.py            dataset exploratory analysis
  ablation_mlp.py         MLP depth / optimiser ablation (retrains from scratch)
  ablation_preprocess.py  MLP preprocessing ablation (3 seeds per variant)
  generate_data.py  optional synthetic fallback generator (NOT used for any reported result)
tests/              pytest suite (models, metric, preprocessing, one train/eval loop)
results/            stored metrics (JSON) and figures for every number in the report
                    (model checkpoints are not included: too large, regenerable with train.py)
```

## How to run and reproduce
```bash
# 1. environment
pip install -r requirements.txt          # or: conda env create -f environment.yml

# 2. data (~1.3 GB download) and preprocessing
python src/download_data.py
python src/preprocess.py

# 3. train the two models (writes results/run_001 = FNO, results/run_002 = MLP)
python src/train.py --config configs/fno.yaml
python src/train.py --config configs/mlp.yaml

# 4. evaluate on the 200 held-out test samples + sample plots
python src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt
python src/evaluate.py --config configs/mlp.yaml --checkpoint results/run_002/best_model.pt

# 5. exploratory analysis and ablations
python src/eda.py
python src/ablation_mlp.py --config configs/mlp.yaml
python src/ablation_preprocess.py

# 6. unit tests
pytest
```

## Results (200 held-out test samples, relative L2 error in physical units)
| Model | Params | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| MLP baseline (3 x 2048) | 42.0 M | 0.0820 | 0.0695 | 0.0456 | 0.0325 | 0.3367 |
| FNO (width 64, 12 modes, 4 blocks) | 4.7 M | 0.0521 | 0.0398 | 0.0451 | 0.0155 | 0.3467 |

The FNO has 37% lower mean error with 9x fewer parameters. Stored per-sample errors:
`results/run_00x/evaluation/eval_metrics.json`.

**MLP depth / optimiser ablation** (`results/ablation_mlp.json`, single run each): 1 layer 0.1008,
2 layers 0.0796, 3 layers (final baseline) 0.0857, 3 layers with SGD+momentum 0.1209.

**MLP preprocessing ablation** (`results/ablation_preprocess.json`, mean over 3 seeds): reference 0.0842,
no coordinate channels 0.0843, no input normalisation 0.0864, no target normalisation 0.0917.

**Dataset EDA** (`results/eda_metrics.json`): kappa is exactly bimodal; high-permeability area fraction
0.48 +/- 0.21; 180/200 test fields have a single connected high-kappa region and 2/200 are exactly uniform.
The worst FNO test sample (index 95, error 0.347) is one of the uniform fields.

## Reproducibility notes
- Seed 42 fixes the data subset and split; `train.py` also seeds Python / NumPy / PyTorch and the batch order.
  The two reported runs (`run_001`, `run_002`) were trained before model-initialisation seeding was added, so
  re-running gives numbers close to but not bit-identical to those above (run-to-run differences of
  about 0.004 in mean error were measured for the MLP; see the ablation scripts, which are seeded).
- Training loss is MSE on standardised pressure. The reported metric (relative L2) is computed after
  converting back to physical units (`physical_rel_l2` in `train.py`).
- The cosine learning-rate schedule (`CosineAnnealingLR`, `T_max` = epochs) is stepped once per mini-batch, so the
  learning rate cycles between 1e-3 and 0 every 200 steps; this is identical for all models and ablations.
- Metrics and figures are stored in `results/`, so the report numbers can be checked without retraining.

## References
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
- Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022.
