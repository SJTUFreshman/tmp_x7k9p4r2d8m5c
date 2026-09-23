#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
validate_common

EVAL_VARIANT="${EVAL_VARIANT:-rl}"
case "$EVAL_VARIANT" in
  base) OUTPUT="$BASE_EVAL_OUTPUT"; SUMMARY="$BASE_EVAL_SUMMARY"; ADAPTER="" ;;
  rl) OUTPUT="$RL_EVAL_OUTPUT"; SUMMARY="$RL_EVAL_SUMMARY"; ADAPTER="$TRAIN_OUT/final" ;;
  *) fatal "EVAL_VARIANT must be base or rl" ;;
esac
[[ "$DRY_RUN" == 1 || "$MOCK" == 1 || -z "$ADAPTER" || -s "$ADAPTER/adapter_model.safetensors" ]] || fatal "missing adapter: $ADAPTER"
mkdir -p "$LOG_DIR/04_sas_eval" "$(dirname "$OUTPUT")"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] strict one-shot SAS $EVAL_VARIANT eval on Conifer test"
  exit 0
fi
if [[ "$RESUME" == 0 ]]; then
  reject_existing "$OUTPUT" "$SUMMARY" "${OUTPUT%.jsonl}_rollout_stats.json" "${OUTPUT%.jsonl}_failures.jsonl"
fi
if [[ "$RESUME" == 1 ]] && jsonl_complete "$OUTPUT" "$TEST_DATA" "$EVAL_START" "$EVAL_LIMIT" 1 && [[ -s "$SUMMARY" ]]; then
  echo "[resume] $EVAL_VARIANT evaluation already complete"
  exit 0
fi

trap 'status=$?; trap - EXIT INT TERM; stop_servers; exit "$status"' EXIT INT TERM
[[ "$MOCK" == 1 ]] || start_replicas "$MODEL_14B" sas_policy "$POLICY_PORT" "$LOG_DIR/04_sas_eval/${EVAL_VARIANT}_server" "$ADAPTER"
mock_args=(); [[ "$MOCK" == 0 ]] || mock_args=(--mock)
resume_args=(--resume); [[ "$RESUME" == 1 ]] || resume_args=(--no-resume)
# Temperature-0 retries are deterministic, so a straggler only clears with a larger
# budget; escalate identically for base and rl so the two stay comparable.
eval_budget="$EVAL_MAX_NEW_TOKENS"
eval_status=1
for ((pass=1; pass<=EVAL_AUTO_RESUME_PASSES; pass++)); do
  keep_failures_args=()
  if (( pass == EVAL_AUTO_RESUME_PASSES )); then
    keep_failures_args=(--keep-protocol-failures)
  fi
  set +e
  run_pipeline --phase rollout --data-path "$TEST_DATA" --start "$EVAL_START" --limit "$EVAL_LIMIT" \
    --num-rollouts 1 --api-base "$(endpoint_pool "$POLICY_PORT")" --api-model sas_policy \
    --temperature "$EVAL_TEMPERATURE" --top-p "$EVAL_TOP_P" --max-new-tokens "$eval_budget" \
    --protocol-retries "$PROTOCOL_RETRIES" --concurrency "$EVAL_CONCURRENCY" --api-timeout "$API_TIMEOUT" \
    --seed "$GENERATION_SEED" --source-policy "Qwen3-14B-${EVAL_VARIANT}-strict-SAS" \
    --output "$OUTPUT" --stats-output "${OUTPUT%.jsonl}_rollout_stats.json" \
    --audit-output "${OUTPUT%.jsonl}_failures.jsonl" "${keep_failures_args[@]}" \
    "${resume_args[@]}" "${mock_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/04_sas_eval/${EVAL_VARIANT}_run.log"
  eval_status=${PIPESTATUS[0]}
  set -e
  [[ "$eval_status" == 0 ]] && break
  resume_args=(--resume)
  eval_budget=$((eval_budget + EVAL_TOKEN_ESCALATION))
  echo "[eval] pass $pass incomplete; retrying missing keys with max_new_tokens=$eval_budget"
done
stop_servers
[[ "$eval_status" == 0 ]] || fatal "$EVAL_VARIANT evaluation incomplete after $EVAL_AUTO_RESUME_PASSES passes"
run_pipeline --phase summarize --input "$OUTPUT" --stats-output "$SUMMARY"
[[ -s "$OUTPUT" && -s "$SUMMARY" ]] || fatal "$EVAL_VARIANT evaluation output missing"
