#!/usr/bin/env bash
# Same 30-epoch, seed-7 replay experiment as the offline box pilot.
set -euo pipefail
cd "$(dirname "$0")"
baseline=${1:-out/experiment8_offline_20261125_091046_37t0gB/bbox}
if [[ ! -s "$baseline/replay_prepost-s7/last.pt" || ! -s "$baseline/summary.json" ]]; then
    echo "Pass the original completed bbox run directory as the first argument." >&2
    exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
mkdir -p out
run_dir=$(mktemp -d "out/experiment8_cutouts_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "Cutout run: $run_dir"
mask_args=()
if [[ -n ${EXPERIMENT8_MASK_WEIGHTS:-} ]]; then
    mask_args+=(--weights "$EXPERIMENT8_MASK_WEIGHTS")
fi
"${EXPERIMENT8_SEGMENT_PYTHON:-python}" -u prepare_experiment8_cutouts.py --output "$run_dir/data" "${mask_args[@]}"
python -u run_memory_replay.py train --data "$run_dir/data" --plan "$run_dir/data/plan.json" \
    --output "$run_dir/foreground" --crop-mode foreground --variant replay_prepost competitive_prepost \
    --device cuda:0 --epochs 30 --seed 7
cat "$run_dir/foreground/report.md" "$run_dir/foreground/writer_report.md"
python -u run_experiment8_followup.py "$run_dir/foreground" --compare-bbox "$baseline"
