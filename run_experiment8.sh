#!/usr/bin/env bash
# Run from the dino-probe environment on the training server.
set -euo pipefail
cd "$(dirname "$0")"
epochs=${1:-30}
if (( $# > 0 )); then shift; fi
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(7); fi
data=${CLEAR_MEMORY_DATA:-data/experiment8_clear_orbits_v1}
capture_python=${CARLA_PYTHON:-python}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

if [[ -n "${MEMORY_PLAN:-}" ]]; then
    echo "Using the clean plan in $data. The legacy MEMORY_PLAN setting is ignored."
fi
if [[ -f "$data/plan.json" ]]; then
    python -u capture_clear_memory.py audit --data "$data"
else
    "$capture_python" -u capture_clear_memory.py capture --data "$data" \
        --host "${CARLA_HOST:-127.0.0.1}" --port "${CARLA_PORT:-2000}"
fi
python -m unittest discover -s tests -p test_writer_competition.py -v
mkdir -p out
out_dir=$(mktemp -d "out/experiment8_clean_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "Experiment 8 output: $out_dir/foreground"
python -u run_memory_replay.py train --data "$data" --plan "$data/plan.json" --output "$out_dir/foreground" \
    --crop-mode foreground --variant replay_prepost competitive_prepost \
    --device cuda:0 --epochs "$epochs" --seed "${seeds[@]}"
cat "$out_dir/foreground/report.md" "$out_dir/foreground/writer_report.md"
