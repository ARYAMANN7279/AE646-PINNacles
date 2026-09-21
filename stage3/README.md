# Stage 3 — Final report, deck, code map

Deliverables: `PINNacles_Stage3_FinalReport.pdf` (8 pages), `PINNacles_Stage3_FinalPresentation.pptx/.pdf` (13 slides),
`PINNacles_Stage3_ContributionStatement.pdf` (contribution statement + AI-tool-use declaration), code in `../src`.

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
(regenerate with `jobs_final.txt`). Timing numbers come from a shared 16-core host with an idle RTX PRO 6000; CPU timings are indicative.
Selection of all hyper-parameters used validation data only; the test set was evaluated once per run.
`stage1/` and `stage2/` are the submitted earlier stages and are unchanged.
