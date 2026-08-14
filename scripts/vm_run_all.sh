#!/bin/bash
# Runs the full AE646 pipeline on the VM (GPU 1 - GPUs 0/2/3 are busy with other users' jobs).
set -e
cd /SML_DISK_24TB/rajeshr/Aryamann/ae646
export CUDA_VISIBLE_DEVICES=1
PY=/SML_DISK_24TB/rajeshr/Aryamann/env/bin/python3

echo "=== Preprocessing ==="
$PY src/preprocess.py

echo "=== Training FNO (original) ==="
mkdir -p results/run_001
$PY src/train.py --config configs/fno.yaml > results/run_001/train.log 2>&1
$PY src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt

echo "=== Training MLP baseline ==="
mkdir -p results/run_002
$PY src/train.py --config configs/mlp.yaml > results/run_002/train.log 2>&1
$PY src/evaluate.py --config configs/mlp.yaml --checkpoint results/run_002/best_model.pt

echo "=== Training FNO (improved) ==="
mkdir -p results/run_003_fno_improved
$PY src/train.py --config configs/fno_improved.yaml > results/run_003_fno_improved/train.log 2>&1
$PY src/evaluate.py --config configs/fno_improved.yaml --checkpoint results/run_003_fno_improved/best_model.pt

echo "=== Zero-shot super-resolution (FNO original + improved) ==="
$PY src/superres_eval.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt --output results/run_001/superres_metrics.json
$PY src/superres_eval.py --config configs/fno_improved.yaml --checkpoint results/run_003_fno_improved/best_model.pt --output results/run_003_fno_improved/superres_metrics.json

echo "=== Inference speed benchmark ==="
$PY src/benchmark_speed.py --fno-config configs/fno.yaml --fno-checkpoint results/run_001/best_model.pt \
  --mlp-config configs/mlp.yaml --mlp-checkpoint results/run_002/best_model.pt \
  --output results/benchmark_speed.json

echo "=== Comparison plots ==="
$PY src/compare_comprehensive.py

echo "VM_PIPELINE_COMPLETE"
