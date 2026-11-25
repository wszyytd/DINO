#!/usr/bin/env bash
# Run from the dino-probe environment on the training server.
set -euo pipefail
cd "$(dirname "$0")"
offline=false
if [[ ${1:-} == --offline ]]; then offline=true; shift; fi
epochs=${1:-30}
if (( $# > 0 )); then shift; fi
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(7); fi
data=${CLEAR_MEMORY_DATA:-data/experiment8_clear_orbits_v1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

if [[ $offline == false && -n "${MEMORY_PLAN:-}" ]]; then
    echo "Using the clean plan in $data. The legacy MEMORY_PLAN setting is ignored."
fi
mkdir -p out
crop=foreground
if [[ $offline == true ]]; then
    crop=bbox
    out_dir=$(mktemp -d "out/experiment8_offline_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    data="$out_dir/data"
    python -u prepare_experiment8_offline.py --output "$data"
elif [[ -f "$data/plan.json" ]]; then
    python -u capture_clear_memory.py audit --data "$data"
else
    capture=( "${CARLA_PYTHON:-python}" )
    if [[ -z "${CARLA_PYTHON:-}" ]] && ! python -c 'import carla' 2>/dev/null; then
        capture=(conda run --no-capture-output -n carla10 python)
    fi
    echo "Capture interpreter: ${capture[*]}"
    "${capture[@]}" -u capture_clear_memory.py capture --data "$data" \
        --host "${CARLA_HOST:-127.0.0.1}" --port "${CARLA_PORT:-2000}"
fi
python -m unittest discover -s tests -p test_writer_competition.py -v
if [[ $offline == false ]]; then
    out_dir=$(mktemp -d "out/experiment8_clean_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
echo "Experiment 8 output: $out_dir/$crop"
python -u run_memory_replay.py train --data "$data" --plan "$data/plan.json" --output "$out_dir/$crop" \
    --crop-mode "$crop" --variant replay_prepost competitive_prepost \
    --device cuda:0 --epochs "$epochs" --seed "${seeds[@]}"
cat "$out_dir/$crop/report.md" "$out_dir/$crop/writer_report.md"
