#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
validate_common

mkdir -p "$RUN_DIR/01_rollout" "$RUN_DIR/02_score"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] rollout: train ${TRAIN_LIMIT} x ${NUM_ROLLOUTS}, non-thinking, 8192 tokens"
  echo "[dry-run] judge: Qwen3-14B thinking, weights ${OUTCOME_WEIGHT}/${PROCESS_WEIGHT}"
  exit 0
fi

trap 'status=$?; trap - EXIT INT TERM; stop_server; exit "$status"' EXIT INT TERM
wait_idle_gpus

echo "========== 1/2 MATH SAS TRAIN ROLLOUT =========="
start_server "$POLICY_PORT" sas_policy "$MAX_MODEL_LEN" "$RUN_DIR/01_rollout/server.log"
rollout_resume=(--resume)
[[ "$RESUME" == "1" ]] || rollout_resume=(--no-resume)
run_pipeline --phase rollout --data-root "$MATH_DATA_ROOT" --split train \
  --subjects "$SUBJECTS" --start "$TRAIN_START" --limit "$TRAIN_LIMIT" \
  --api-base "http://127.0.0.1:${POLICY_PORT}/v1" --api-model sas_policy \
  --temperature "$ROLLOUT_TEMPERATURE" --top-p "$ROLLOUT_TOP_P" \
  --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS" --no-enable-thinking --no-require-thinking \
  --protocol-retries "$PROTOCOL_RETRIES" --seed "$GENERATION_SEED" \
  --num-rollouts "$NUM_ROLLOUTS" --concurrency "$ROLLOUT_CONCURRENCY" \
  --api-timeout "$API_TIMEOUT" --output "$RAW_ROLLOUT" --stats-output "$ROLLOUT_STATS" \
  "${rollout_resume[@]}" 2>&1 | tee -a "$RUN_DIR/01_rollout/run.log"
stop_server

echo "========== 2/2 MATH SAS QWEN14B JUDGE =========="
wait_idle_gpus
start_server "$JUDGE_PORT" qwen14b_math_sas_judge "$JUDGE_MAX_MODEL_LEN" "$RUN_DIR/02_score/server.log"
judge_resume=(--resume)
[[ "$RESUME" == "1" ]] || judge_resume=(--no-resume)
judge_failures=-1
for ((pass=1; pass<=JUDGE_AUTO_RESUME_PASSES; pass++)); do
  echo "[judge] pass ${pass}/${JUDGE_AUTO_RESUME_PASSES}"
  run_pipeline --phase judge --data-root "$MATH_DATA_ROOT" --split train \
    --subjects "$SUBJECTS" --start "$TRAIN_START" --limit "$TRAIN_LIMIT" \
    --input "$RAW_ROLLOUT" --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge-model qwen14b_math_sas_judge --judge-temperature "$JUDGE_TEMPERATURE" \
    --judge-top-p "$JUDGE_TOP_P" --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
    --judge-enable-thinking --judge-retries "$JUDGE_RETRIES" \
    --outcome-weight "$OUTCOME_WEIGHT" --process-weight "$PROCESS_WEIGHT" \
    --concurrency "$JUDGE_CONCURRENCY" --judge-timeout "$API_TIMEOUT" \
    --seed "$GENERATION_SEED" --output "$JUDGED_ROLLOUT" \
    --audit-output "$JUDGE_AUDIT" --stats-output "$JUDGE_STATS" \
    "${judge_resume[@]}" 2>&1 | tee -a "$RUN_DIR/02_score/run.log"
  judge_resume=(--resume)
  judge_failures="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["judge_failures"])' "$JUDGE_STATS")"
  [[ "$judge_failures" == "0" ]] && break
done
stop_server

[[ -s "$JUDGED_ROLLOUT" ]] || fatal "judge produced no trainable rows"
if [[ "$judge_failures" != "0" ]]; then
  echo "[warning] judge failures after auto-resume passes: $judge_failures"
fi
echo "[done] rollout=$RAW_ROLLOUT judged=$JUDGED_ROLLOUT"
