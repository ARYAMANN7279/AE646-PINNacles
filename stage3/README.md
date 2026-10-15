# Stage 3 — Final report, deck, code map

Deliverables: `PINNacles_Stage3_FinalReport.pdf` (8 pages), `PINNacles_Stage3_FinalPresentation.pptx/.pdf` (13 slides),
`PINNacles_Stage3_ContributionStatement.pdf` (one page: contribution statement + AI-tool-use declaration), code in `../src`.

## Run instructions (Google Colab notebook)

The complete project (data download -> Stage 1/2 baselines -> ablations -> unit tests -> Stage 3 live demos -> stored Stage 3 study) is one notebook,
`PINNacles_Stage3_Notebook.ipynb` (also inside `PINNacles_Stage3_Code.zip`). To run it on Google Colab:
1. Open it: `https://colab.research.google.com/github/ARYAMANN7279/AE646-PINNacles/blob/main/stage3/PINNacles_Stage3_Notebook.ipynb`
   (or upload the notebook from the zip to Colab).
2. `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`. Click "Run anyway" on Colab's standard "not authored by Google" notice.
3. The first cell installs only the packages Colab does not already have (`python-pptx`, `pyamg`); the notebook downloads the PDEBench file itself
   (`2D_DarcyFlow_beta1.0_Train.hdf5`, ~1.3 GB, MD5-checked), writes all sources to disk, and runs every cell. No other setup is needed.

**Computational environment.** The notebook was executed end to end, with no errors, on a Google Colab T4 GPU runtime (about 50 minutes) and also on a
Kaggle T4x2 GPU notebook and a local CUDA GPU workstation. Live results (baselines FNO 0.0528 / MLP ~0.083 mean relative L2, 16/16 unit tests, finite-volume
solver errors 0.0939 / 0.0438 / 0.0241) reproduce the reported numbers; a fresh training run is close to, but not bit-identical to, the stored one (GPU non-determinism).
Without a GPU the training cells are much slower (hours). Python 3 with PyTorch 2.x, NumPy, SciPy, h5py (see `requirements.txt` for the exact versions of the
original environment). Seeds are fixed (42 for the data split).

The full Stage 3 study (142 training runs: recipe, architecture, data-scaling and resolution studies, five-seed final models, latency, diagnostics) took many GPU-hours and was run on a
GPU workstation with `scripts/stage3_run_all.sh`; its results are shipped in `results/stage3/` and loaded (not recomputed) by Section 10 of the notebook. The notebook says so explicitly.

Reproduce everything: `bash scripts/stage3_run_all.sh` (from the repo root; see the script header).
Compile: `tectonic PINNacles_Stage3_FinalReport.tex`; rebuild the deck: `python3 build_stage3_presentation.py`.

| Report item | Produced by | Data |
|---|---|---|
| Table 1, Fig. 1 (recipe) | `stage3_train.py` via `scripts/stage3_jobs/jobs_e1.txt`; `stage3_aggregate.py` | `results/stage3/runs/e1_*` |
| Fig. 2, Table 2 (architecture) | `jobs_e2.txt`, `jobs_e5.txt` | `runs/e2_*`, `runs/e5_*` |
| Table 3 (final models) | `jobs_final.txt` (5 seeds each) | `runs/final_*` |
| Fig. 3 (scaling, resolution) | `jobs_e34.txt` | `runs/e3_*`, `runs/e4_*` |
| Table 4 (solver) | `solver_vs_fno.py` (uses `solvers.py`) | `solver_vs_gt.json` |
| Table 5, Fig. 4 (latency) | `stage3_latency.py` | `latency.json`, `pareto.json` |
| Figs. 5–6, §8 numbers | `stage3_diagnostics.py` | `diagnostics.json`, `diagnostics_arrays.npz` |

Notes: `results/stage3/runs/*/metrics.json` hold the per-sample test errors and full configs; model checkpoints (`best_model.pt`) are not tracked
(regenerate with `jobs_final.txt`). Timing numbers come from the GPU workstation (idle GPU, 16-core CPU); CPU timings are indicative.
Selection of all hyper-parameters used validation data only; the test set was evaluated once per run.
`stage1/` and `stage2/` are the submitted earlier stages and are unchanged.

Trained checkpoints are not tracked (git-ignored); regenerate them with `jobs_final.txt`.
