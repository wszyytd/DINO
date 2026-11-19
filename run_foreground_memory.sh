#!/usr/bin/env bash
# Run on the training server from the existing dino-probe environment.
set -euo pipefail
cd "$(dirname "$0")"
epochs=${1:-30}
if (( $# > 0 )); then shift; fi
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(7); fi
plan=${MEMORY_PLAN:-out/memory_retention_20261117_110524/plan.json}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

python run_memory_replay.py audit --plan "$plan"
RUN_MEMORY_REPLAY_CUDA_TESTS=1 python -m unittest discover -s tests -p test_memory_replay_server.py -v
mkdir -p out
run_dir=$(mktemp -d "out/foreground_memory_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "Experiment output: $run_dir"
common=(--plan "$plan" --epochs "$epochs" --seed "${seeds[@]}" --device cuda:0)
python -u run_memory_replay.py train "${common[@]}" --output "$run_dir/bbox" \
    --crop-mode bbox --variant replay_independent
python -u run_memory_replay.py train "${common[@]}" --output "$run_dir/foreground" \
    --crop-mode foreground --variant replay_independent replay_prepost ema_prepost
cat "$run_dir/bbox/report.md" "$run_dir/foreground/report.md"
