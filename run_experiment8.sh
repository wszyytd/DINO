#!/usr/bin/env bash
# Run from the dino-probe environment on the training server.
set -euo pipefail
cd "$(dirname "$0")"
epochs=${1:-30}
if (( $# > 0 )); then shift; fi
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(7); fi
plan=${MEMORY_PLAN:-out/memory_retention_20261117_110524/plan.json}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

python run_memory_replay.py audit --plan "$plan"
python -m unittest discover -s tests -p test_writer_competition.py -v
mkdir -p out
out_dir=$(mktemp -d "out/experiment8_writer_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "Experiment 8 output: $out_dir/foreground"
python -u run_memory_replay.py train --plan "$plan" --output "$out_dir/foreground" \
    --crop-mode foreground --variant replay_prepost competitive_prepost \
    --device cuda:0 --epochs "$epochs" --seed "${seeds[@]}"
cat "$out_dir/foreground/report.md" "$out_dir/foreground/writer_report.md"
