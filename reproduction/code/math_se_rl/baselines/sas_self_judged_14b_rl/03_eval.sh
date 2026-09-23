#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
validate_common

EVAL_VARIANT="${EVAL_VARIANT:-rl}"
EVAL_TAG="${EVAL_TAG:-$EVAL_VARIANT}"
case "$EVAL_VARIANT" in
  base)
    OUTPUT="${EVAL_OUTPUT:-$BASE_EVAL_OUTPUT}"
    STATS="${EVAL_STATS:-$BASE_EVAL_STATS}"
    SERVER_MODEL_NAME="sas_policy"
    API_MODEL_NAME="sas_policy"
    adapter_args=()
    ;;
  rl)
    OUTPUT="${EVAL_OUTPUT:-$RL_EVAL_OUTPUT}"
    STATS="${EVAL_STATS:-$RL_EVAL_STATS}"
    SERVER_MODEL_NAME="sas_base"
    API_MODEL_NAME="sas_policy"
    ADAPTER="${EVAL_ADAPTER:-$TRAIN_OUT/final}"
    [[ "$DRY_RUN" == "1" || -s "$ADAPTER/adapter_model.safetensors" ]] || \
      fatal "missing trained adapter: $ADAPTER"
    adapter_args=(--enable-lora --max-lora-rank "$LORA_RANK" --max-loras 1 \
      --max-cpu-loras 1 --lora-dtype auto --lora-modules "sas_policy=$ADAPTER")
    ;;
  *) fatal "EVAL_VARIANT must be base or rl" ;;
esac

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] eval variant=$EVAL_VARIANT test=$EVAL_LIMIT ids=${EVAL_PROBLEM_IDS_FILE:-contiguous} thinking=$EVAL_ENABLE_THINKING/$EVAL_REQUIRE_THINKING"
  echo "[dry-run] max_tokens=$EVAL_MAX_NEW_TOKENS max_model_len=$MAX_MODEL_LEN"
  exit 0
fi

  mkdir -p "$RUN_DIR/04_eval"
trap 'status=$?; trap - EXIT INT TERM; stop_server; exit "$status"' EXIT INT TERM
wait_idle_gpus
  start_server "$POLICY_PORT" "$SERVER_MODEL_NAME" "$MAX_MODEL_LEN" \
  "$RUN_DIR/04_eval/${EVAL_TAG}_server.log" "${adapter_args[@]}"
if [[ "$EVAL_VARIANT" == "rl" ]]; then
  wait_server "$POLICY_PORT" "$API_MODEL_NAME"
fi

resume_args=(--resume)
[[ "$RESUME" == "1" ]] || resume_args=(--no-resume)
problem_ids_args=()
[[ -z "$EVAL_PROBLEM_IDS_FILE" ]] || problem_ids_args=(--problem-ids-file "$EVAL_PROBLEM_IDS_FILE")
thinking_args=(--no-enable-thinking --no-require-thinking)
[[ "$EVAL_ENABLE_THINKING" == "1" ]] && thinking_args[0]=--enable-thinking
[[ "$EVAL_REQUIRE_THINKING" == "1" ]] && thinking_args[1]=--require-thinking
run_pipeline --phase rollout --data-root "$MATH_DATA_ROOT" --split test \
  --subjects "$SUBJECTS" --start "$EVAL_START" --limit "$EVAL_LIMIT" \
  "${problem_ids_args[@]}" \
  --api-base "http://127.0.0.1:${POLICY_PORT}/v1" --api-model "$API_MODEL_NAME" \
  --temperature "$EVAL_TEMPERATURE" --top-p "$EVAL_TOP_P" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" "${thinking_args[@]}" \
  --protocol-retries "$PROTOCOL_RETRIES" --seed "$GENERATION_SEED" \
  --num-rollouts 1 --concurrency "$EVAL_CONCURRENCY" --api-timeout "$API_TIMEOUT" \
  --output "$OUTPUT" --stats-output "$STATS" "${resume_args[@]}" \
  2>&1 | tee -a "$RUN_DIR/04_eval/${EVAL_TAG}_run.log"
stop_server

[[ -s "$OUTPUT" && -s "$STATS" ]] || fatal "evaluation output missing"
echo "[done] variant=$EVAL_VARIANT output=$OUTPUT summary=$STATS"
