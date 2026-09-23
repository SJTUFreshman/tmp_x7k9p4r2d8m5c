#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

COLLECTOR="${EXPERIMENT_ROOT}/collect_math_correction_sft_rollouts.py"
LAUNCHER="${PROJECT_ROOT}/scripts/run_math_correction_sft_rollout_3x8b_8gpu.sh"
BUILDER="${EXPERIMENT_ROOT}/build_math_correction_sft_from_rollout.py"
require_file "$COLLECTOR"
require_file "$LAUNCHER"
require_file "$BUILDER"

VALIDATE_ROLLOUT_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$COLLECTOR"
  --data-root "$MATH_DATA_ROOT" --start "$SFT_ROLLOUT_START"
  --limit "$SFT_ROLLOUT_LIMIT" --plan-offset "$SFT_ROLLOUT_PLAN_OFFSET"
  --max-missing "$SFT_ROLLOUT_MAX_MISSING"
  --output "$SFT_ROLLOUT_FILE" --validate-existing
)
ROLLOUT_CMD=(
  env
  "PROJECT_ROOT=$PROJECT_ROOT" "PYTHONPATH_ROOT=$PYTHONPATH_ROOT"
  "PYTHON_BIN=$PYTHON_BIN" "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN"
  "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH"
  "COLLECTOR=$COLLECTOR" "DATA_ROOT=$MATH_DATA_ROOT"
  "START=$SFT_ROLLOUT_START" "LIMIT=$SFT_ROLLOUT_LIMIT"
  "MAX_MISSING=$SFT_ROLLOUT_MAX_MISSING"
  "PLAN_OFFSET=$SFT_ROLLOUT_PLAN_OFFSET" "OUTPUT_PATH=$SFT_ROLLOUT_FILE"
  "MODEL_8B=$SFT_ROLLOUT_MODEL"
  "MAX_CONCURRENCY=$SFT_ROLLOUT_MAX_CONCURRENCY"
  "MAX_NEW_TOKENS=$SFT_ROLLOUT_MAX_NEW_TOKENS"
  "TEMPERATURE=$SFT_ROLLOUT_TEMPERATURE" "TOP_P=$SFT_ROLLOUT_TOP_P"
  "MAX_STEP_RETRIES=$SFT_ROLLOUT_MAX_STEP_RETRIES"
  "MAX_TRAJECTORY_ATTEMPTS=$SFT_ROLLOUT_MAX_TRAJECTORY_ATTEMPTS"
  "MAX_VERIFIER_SIMILARITY=$SFT_ROLLOUT_MAX_VERIFIER_SIMILARITY"
  "GENERATION_SEED=$SFT_SEED" "LOG_DIR=$SFT_ROLLOUT_LOG_ROOT"
  "RUN_ID=${TAG}_sft_rollout" "RESUME=$RESUME" "DRY_RUN=$DRY_RUN"
  bash "$LAUNCHER"
)
BUILD_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$BUILDER"
  --input "$SFT_ROLLOUT_FILE" --out-dir "$SFT_DATA_DIR"
  --samples-per-agent "$SFT_SAMPLES_PER_AGENT"
  --correction-fraction "$SFT_CORRECTION_FRACTION"
  --balance-seed "$SFT_SEED"
  --max-verifier-reasoning-similarity "$SFT_ROLLOUT_MAX_VERIFIER_SIMILARITY"
)

echo "MATH-specific fixed-A1 SFT data"
echo "  teachers:    Qwen3-8B x 3 base models on MATH train"
echo "  source:      accepted controlled model rollouts only"
echo "  routes:      A1>A2>A3 and A1>A3>A2 (50/50)"
echo "  A1:          correct solver/handoff labels only"
echo "  A2/A3:       70% regular + 30% real correction, 50% confirm"
echo "  rollout:     $SFT_ROLLOUT_FILE"
echo "  coverage:    allow at most $SFT_ROLLOUT_MAX_MISSING missing plans; row quality unchanged"
echo "  output:      $SFT_DATA_DIR"
print_command "${ROLLOUT_CMD[@]}"
print_command "${BUILD_CMD[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  "${ROLLOUT_CMD[@]}"
  exit 0
fi

if [[ -s "$SFT_DATA_DIR/stats.json" ]]; then
  "${VALIDATE_ROLLOUT_CMD[@]}"
  "${BUILD_CMD[@]}" --validate-existing
  echo "[resume] validated rollout-derived MATH SFT data"
  exit 0
fi
if [[ -e "$SFT_DATA_DIR" ]]; then
  fatal "partial SFT data directory exists; archive it or choose a new TAG: $SFT_DATA_DIR"
fi

if ! "${VALIDATE_ROLLOUT_CMD[@]}" >/dev/null 2>&1; then
  acquire_gpu_lock
  for ((attempt=1; attempt<=SFT_ROLLOUT_LAUNCH_ATTEMPTS; attempt++)); do
    log "MATH SFT rollout launcher attempt $attempt/$SFT_ROLLOUT_LAUNCH_ATTEMPTS"
    if "${ROLLOUT_CMD[@]}" && "${VALIDATE_ROLLOUT_CMD[@]}"; then
      break
    fi
    (( attempt < SFT_ROLLOUT_LAUNCH_ATTEMPTS )) || \
      fatal "MATH SFT rollout failed after $SFT_ROLLOUT_LAUNCH_ATTEMPTS attempts"
    sleep 30
  done
else
  log "validated accepted MATH SFT rollout coverage; skipping model servers"
fi

"${BUILD_CMD[@]}"
"${BUILD_CMD[@]}" --validate-existing
echo "[done] rollout-derived MATH SFT data passed all role and source checks"
