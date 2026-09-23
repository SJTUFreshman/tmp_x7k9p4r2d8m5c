#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

RUNNER="${UPSTREAM_MATH_ROOT}/math_role_batched.py"
LAUNCHER="${PROJECT_ROOT}/scripts/run_math_mas_role_batched_8gpu.sh"
require_file "$RUNNER"
require_file "$LAUNCHER"
if [[ "$DRY_RUN" != "1" ]]; then
  require_adapter "$SFT_A1"
  require_adapter "$SFT_A2"
  require_adapter "$SFT_A3"
fi

sample_resume="$RESUME"
if [[ "$RESUME" == "1" && ! -e "$SAMPLE_STATE" && ! -e "$RAW_ROLLOUT" ]]; then
  sample_resume=0
fi
SAMPLE_CMD=(
  env
  "PROJECT_ROOT=$PROJECT_ROOT" "PYTHON_BIN=$PYTHON_BIN"
  "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN"
  "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH"
  "RUNNER=$RUNNER" "DATA_ROOT=$MATH_DATA_ROOT" "SPLIT=train"
  "SUBJECTS=$SUBJECTS" "START=$TRAIN_START" "LIMIT=$TRAIN_LIMIT"
  "NUM_ROLLOUTS=$NUM_ROLLOUTS" "T_MAX=$T_MAX"
  "START_AGENT=$SAMPLE_START_AGENT" "START_AGENT_SEED=$SEED"
  "MIN_AGENTS_BEFORE_STOP=$MIN_AGENTS_BEFORE_STOP" "ALLOW_FIRST_TURN_STOP=0"
  "ENABLE_THINKING=$SAMPLE_ENABLE_THINKING"
  "REQUIRE_THINKING=$SAMPLE_REQUIRE_THINKING" "OUTPUT_MODE=turns"
  "GENERATION_SEED=$SAMPLE_GENERATION_SEED"
  "GROUP_RETRIES=$SAMPLE_GROUP_RETRIES" "STEP_RETRIES=$SAMPLE_STEP_RETRIES"
  "RETRY_FAILED_GROUPS=1" "MAX_NEW_TOKENS=$SAMPLE_MAX_NEW_TOKENS"
  "TEMPERATURE=$SAMPLE_TEMPERATURE" "TOP_P=$SAMPLE_TOP_P"
  "MAX_CONCURRENCY=$SAMPLE_MAX_CONCURRENCY" "API_TIMEOUT=$API_TIMEOUT"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "JSON_TRANSPORT=$SAMPLE_JSON_TRANSPORT"
  "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
  "USE_LORA=1" "ADAPTER_A1=$SFT_A1" "ADAPTER_A2=$SFT_A2" "ADAPTER_A3=$SFT_A3"
  "OUTPUT_PATH=$RAW_ROLLOUT" "OUTPUT_DIR=$(dirname "$RAW_ROLLOUT")"
  "STATE_PATH=$SAMPLE_STATE" "RUN_ID=${TAG}_02_sample_math_sft"
  "RUN_DIR=${LOG_ROOT}/02_sample" "LOG_DIR=${LOG_ROOT}/02_sample"
  "RESUME=$sample_resume" "KEEP_SERVERS=0"
  bash "$LAUNCHER"
)

echo "Fresh MATH rollout from MATH-specific SFT"
echo "  policy:      ${SFT_RUN_ROOT}/{A1,A2,A3}/final"
echo "  source:      MATH train (${TRAIN_LIMIT} problems)"
echo "  coverage:    ${NUM_ROLLOUTS} rollouts/problem"
echo "  output:      $RAW_ROLLOUT"
print_command "${SAMPLE_CMD[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] rollout not started"
  exit 0
fi

if [[ "$RESUME" == "1" && -e "$RAW_ROLLOUT" ]]; then
  require_file "$SAMPLE_STATE"
  finalize_args=()
  if [[ -n "${MATH_FAILURE_LEDGER:-}" ]]; then
    [[ "$MATH_FAILURE_LEDGER" == "${RAW_ROLLOUT}.failures.jsonl" ]] || fatal "failure ledger path differs from native sidecar"
    [[ "${MATH_SAMPLE_STATE:-}" == "$SAMPLE_STATE" ]] || fatal "failure ledger state differs from sampling state"
    finalize_args+=(--allow-zero-step-failures)
  fi
  env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" finalize \
    --state "$SAMPLE_STATE" --output "$RAW_ROLLOUT" --resume "${finalize_args[@]}"
  if [[ -n "${MATH_FAILURE_LEDGER:-}" ]]; then
    "$PYTHON_BIN" - "$RAW_ROLLOUT" "$MATH_FAILURE_LEDGER" "$MATH_SAMPLE_STATE" <<'PY'
import collections
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.environ["PYTHONPATH_ROOT"])
from jca.experiments.math_rl_mas_thinking.sampling_coverage import validate_sampling_coverage
groups = collections.defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            groups[(str(row["problem_id"]), int(row["rollout_idx"]))].append(row)
coverage = validate_sampling_coverage(
    groups, expected_rollouts=8,
    data_root=Path(os.environ["MATH_DATA_ROOT"]),
    failure_ledger=Path(sys.argv[2]), sample_state=Path(sys.argv[3]),
)
print("[resume] rollout has validated zero-step failure sidecar:", coverage)
PY
  fi
  echo "[resume] rollout already complete and atomically validated"
  exit 0
fi
if [[ "$RESUME" != "1" && ( -e "$SAMPLE_STATE" || -e "$RAW_ROLLOUT" ) ]]; then
  fatal "rollout output/state exists and RESUME=0"
fi

mkdir -p "$(dirname "$RAW_ROLLOUT")" "$LOG_ROOT/02_sample"
acquire_gpu_lock
wait_for_idle_gpus "fresh MATH-SFT rollout"
exec "${SAMPLE_CMD[@]}"
