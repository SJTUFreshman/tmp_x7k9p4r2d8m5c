#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

STAGE="${UPSTREAM_MATH_ROOT}/02_score.sh"
REUSE_TOOL="${EXPERIMENT_ROOT}/reuse_math_judged_scores.py"
require_file "$STAGE"
if [[ "$DRY_RUN" != "1" ]]; then
  require_file "$RAW_ROLLOUT"
  require_adapter "$SFT_A1"
  require_adapter "$SFT_A2"
  require_adapter "$SFT_A3"
fi

if [[ "$REUSE_JUDGE_SCORES" == "1" ]]; then
  require_file "$REUSE_TOOL"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$REUSED_JUDGED_SOURCE"
  fi
  REUSE_CMD=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$REUSE_TOOL"
    --source "$RAW_ROLLOUT" --legacy-judged "$REUSED_JUDGED_SOURCE"
    --output "$JUDGED_ROLLOUT" --stats-output "$JUDGE_STATS"
    --reuse-stats-output "$JUDGE_REUSE_STATS" --data-root "$MATH_DATA_ROOT"
    --expected-rollouts "$NUM_ROLLOUTS" --expected-judge-model "$JUDGE_SERVED_MODEL_NAME"
  )
  echo "Current-contract MATH rollout score reuse"
  echo "  source:         $RAW_ROLLOUT"
  echo "  legacy judged:  $REUSED_JUDGED_SOURCE"
  echo "  output:         $JUDGED_ROLLOUT"
  echo "  safety:         immutable source + deterministic state + current validator"
  print_command "${REUSE_CMD[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] judged-score reuse not started"
    exit 0
  fi
  if [[ "$RESUME" == "1" && -s "$JUDGED_ROLLOUT" && -s "$JUDGE_STATS" && \
        -s "$JUDGE_REUSE_STATS" ]]; then
    "${REUSE_CMD[@]}" --validate-existing
    echo "[resume] reused judge scores are complete and validated"
    exit 0
  fi
  if [[ -e "$JUDGED_ROLLOUT" || -e "$JUDGE_STATS" || -e "$JUDGE_REUSE_STATS" ]]; then
    archive_root="$(dirname "$JUDGED_ROLLOUT")/_superseded_full_rejudge/$(date +%Y%m%d_%H%M%S).$$"
    mkdir -p "$archive_root"
    for path in "$JUDGED_ROLLOUT" "$JUDGE_STATS" "$JUDGE_REUSE_STATS"; do
      [[ -e "$path" ]] || continue
      mv -- "$path" "$archive_root/$(basename "$path")"
    done
    log "archived partial full-rejudge outputs: $archive_root"
  fi
  "${REUSE_CMD[@]}"
  "${REUSE_CMD[@]}" --validate-existing
  echo "[done] historical Qwen14B process scores reused under the current artifact contract"
  exit 0
fi

SCORE_CMD=(
  env
  "EXPERIMENT_ROOT=$UPSTREAM_MATH_ROOT" "PROJECT_ROOT=$PROJECT_ROOT"
  "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "PYTHON_BIN=$PYTHON_BIN"
  "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN"
  "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH"
  "TAG=${TAG}_current_rejudge" "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "RAW_ROLLOUT=$RAW_ROLLOUT" "JUDGED_ROLLOUT=$JUDGED_ROLLOUT"
  "JUDGE_STATS=$JUDGE_STATS" "LOG_ROOT=$LOG_ROOT"
  "MATH_DATA_ROOT=$MATH_DATA_ROOT" "TRAIN_START=$TRAIN_START" "TRAIN_LIMIT=$TRAIN_LIMIT"
  "EVAL_START=$EVAL_START" "EVAL_LIMIT=$EVAL_LIMIT" "SUBJECTS=$SUBJECTS"
  "NUM_ROLLOUTS=$NUM_ROLLOUTS" "T_MAX=$T_MAX"
  "SAMPLE_START_AGENT=$SAMPLE_START_AGENT"
  "SAMPLE_ENABLE_THINKING=$SAMPLE_ENABLE_THINKING"
  "SAMPLE_REQUIRE_THINKING=$SAMPLE_REQUIRE_THINKING"
  "TRAIN_ENABLE_THINKING=$TRAIN_ENABLE_THINKING"
  "ENABLE_THINKING=$ENABLE_THINKING" "REQUIRE_THINKING=$REQUIRE_THINKING"
  "ALLOW_THINKING_MODE_MISMATCH=$ALLOW_THINKING_MODE_MISMATCH"
  "SAMPLE_JSON_TRANSPORT=$SAMPLE_JSON_TRANSPORT"
  "SAMPLE_MAX_NEW_TOKENS=$SAMPLE_MAX_NEW_TOKENS"
  "SAMPLE_TEMPERATURE=$SAMPLE_TEMPERATURE" "SAMPLE_TOP_P=$SAMPLE_TOP_P"
  "JUDGE_MODEL_PATH=$JUDGE_MODEL_PATH" "JUDGE_MAX_MODEL_LEN=$JUDGE_MAX_MODEL_LEN"
  "JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS" "JUDGE_CONCURRENCY=$JUDGE_CONCURRENCY"
  "JUDGE_GROUP_RETRIES=$JUDGE_GROUP_RETRIES"
  "JUDGE_PARSE_RETRIES=$JUDGE_PARSE_RETRIES"
  "JUDGE_AUTO_RESUME_PASSES=$JUDGE_AUTO_RESUME_PASSES"
  "JUDGE_TEMPERATURE=$JUDGE_TEMPERATURE" "JUDGE_TOP_P=$JUDGE_TOP_P"
  "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
  "SFT_ROOT=$SFT_RUN_ROOT" "SFT_A1=$SFT_A1" "SFT_A2=$SFT_A2" "SFT_A3=$SFT_A3"
  "RESUME=$RESUME" "DRY_RUN=$DRY_RUN" "WAIT_FOR_GPUS=$WAIT_FOR_GPUS"
  "GPU_POLL_SECONDS=$GPU_POLL_SECONDS" "GPU_IDLE_CHECKS=$GPU_IDLE_CHECKS"
  bash "$STAGE"
)

echo "Current-contract MATH rollout scoring"
echo "  source: $RAW_ROLLOUT"
echo "  judge:  $JUDGE_MODEL_PATH (thinking enabled for judge only)"
echo "  output: $JUDGED_ROLLOUT"
print_command "${SCORE_CMD[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] scoring not started; future rollout and adapters are not required"
  exit 0
fi
acquire_gpu_lock
exec "${SCORE_CMD[@]}"
