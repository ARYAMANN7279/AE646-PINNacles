# AE646 Stage 2 - Fourier Neural Operator for Parametric Darcy Flow

**Team PINNacles** | AE646: Scientific Machine Learning for Fluid Mechanics | Project theme 8 (operator learning for parametric PDEs)

This is the Stage 2 (interim) code: the data pipeline, an MLP baseline (with ablations), and an initial
Fourier Neural Operator (FNO) with preliminary results.

## Problem
Learn the solution operator of the 2D parametric Darcy flow equation

    -div( kappa(x,y) * grad u(x,y) ) = f(x,y)  on (0,1)^2,   u = 0 on the boundary,   f = 1

where kappa is a piecewise-constant permeability field (values 0.1 or 1.0), i.e. learn the map kappa -> u.
Baseline: fully-connected MLP. SciML model: FNO.

## Computational environment
Plain Python scripts (no notebooks). Code development and unit tests were run on a MacBook (macOS, Apple Silicon);
model training and the ablations were run on a GPU workstation (NVIDIA RTX PRO 6000, CUDA). Python 3.11, PyTorch 2.x.

## Dataset
Real PDEBench 2D Darcy Flow (beta = 1.0), DOI 10.18419/darus-2986: 10,000 samples at 128x128 (about 1.3 GB).
From it we draw a reproducible subset (seed 42): 1,000 train/validation samples (split 900 / 100) and 200 held-out
test samples. Fields are subsampled to 64x64 (every second grid point), standardised with training-set statistics
only, and (x, y) coordinate channels are added, giving inputs of shape 64x64x3 = [kappa, x, y].

## Setup and how to run
Unzip the code archive and run everything from the extracted folder.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download the data (about 1.3 GB) and preprocess it
python src/download_data.py
python src/preprocess.py

# 3. train the FNO and the MLP baseline
python src/train.py --config configs/fno.yaml
python src/train.py --config configs/mlp.yaml

# 4. evaluate on the 200 test samples (also saves sample plots)
python src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt
python src/evaluate.py --config configs/mlp.yaml --checkpoint results/run_002/best_model.pt

# 5. dataset analysis and the two ablations
python src/eda.py
python src/ablation_mlp.py --config configs/mlp.yaml
python src/ablation_preprocess.py

# 6. unit tests
pytest
```
Metrics and figures for every number in the report are already stored in the `results` folder of the archive,
so they can be checked without retraining (trained checkpoints are not included because of their size).

## Results (200 held-out test samples, relative L2 error in physical units)
| Model | Params | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| MLP baseline (3 x 2048) | 42.0 M | 0.0820 | 0.0695 | 0.0456 | 0.0325 | 0.3367 |
| FNO (width 64, 12 modes, 4 blocks) | 4.7 M | 0.0521 | 0.0398 | 0.0451 | 0.0155 | 0.3467 |

The FNO reaches 37% lower mean error with 9x fewer parameters.

- **MLP depth / optimiser ablation** (one run each): 1 layer 0.1008, 2 layers 0.0796, 3 layers (final baseline) 0.0857,
  3 layers with SGD+momentum 0.1209.
- **MLP preprocessing ablation** (mean of 3 seeds): reference 0.0842, no coordinate channels 0.0843,
  no input normalisation 0.0864, no target normalisation 0.0917.
- **Dataset analysis**: kappa is exactly bimodal; the high-permeability area fraction is 0.48 +/- 0.21; 180 of the 200
  test fields contain a single high-permeability region and 2 are exactly uniform. The worst FNO test error (0.347)
  is one of the uniform fields.

## Notes
- The random seed is fixed to 42 (data split, model initialisation, batch order). The two reported training runs were
  made before model-initialisation seeding was added, so re-running gives numbers close to, but not exactly equal to,
  the ones above (differences of about 0.004 in mean error were observed).
- The training loss is mean-squared error on standardised pressure; the reported relative L2 error is computed after
  converting back to physical units.

## References
- Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
- M. Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022.
