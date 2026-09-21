#!/bin/bash
# Reproduces every Stage 3 result. Needs the PDEBench file (python src/download_data.py) and a CUDA GPU.
# ~200 runs; with 3 GPUs x 4 concurrent jobs the whole matrix takes a few hours.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
GPUS=${GPUS:-"0"}            # e.g. GPUS="1 2 3"
PER_GPU=${PER_GPU:-4}

$PY src/preprocess.py                       # processed arrays + stats (Stage 2)
$PY src/data_cache.py                       # memory-mappable 128x128 cache + split indices (verified against processed/)
for j in e1 e2 e34 e5 final; do             # recipe, architecture, data+resolution, candidates, 5-seed finals
  $PY src/run_jobs.py scripts/stage3_jobs/jobs_$j.txt --gpus $GPUS --per-gpu $PER_GPU
done
# timing / physics: run on an otherwise idle machine
(cd src && OMP_NUM_THREADS=1 $PY solver_vs_fno.py)                       # solver calibration + timing
(cd src && $PY stage3_latency.py)                                        # CUDA-event latency
(cd src && $PY stage3_diagnostics.py --ckpt ../results/stage3/runs/final_fno_s0/best_model.pt)
(cd src && $PY stage3_aggregate.py && $PY stage3_figures.py)
