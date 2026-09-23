#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
validate_common
mkdir -p "$LOG_DIR/01_sas_rollout" "$LOG_DIR/02_sas_judge" "$LOG_DIR/03_train"

if [[ "$RESUME" == 0 ]]; then
  reject_existing "$RAW_ROLLOUT" "$ROLLOUT_STATS" "$JUDGED_ROLLOUT" "$JUDGE_AUDIT" "$JUDGE_STATS" "$TRAIN_OUT"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] 14B SAS rollout: train limit=${TRAIN_LIMIT:-all} x $NUM_ROLLOUTS"
  echo "[dry-run] 14B self-judge, then fresh-LoRA signed RWR for $NUM_EPOCHS epoch(s)"
  exit 0
fi

trap 'status=$?; trap - EXIT INT TERM; stop_servers; exit "$status"' EXIT INT TERM
mock_args=(); [[ "$MOCK" == 0 ]] || mock_args=(--mock)

echo "========== 1/3 CONIFER 14B SAS ROLLOUT =========="
[[ "$MOCK" == 1 ]] || start_replicas "$MODEL_14B" sas_policy "$POLICY_PORT" "$LOG_DIR/01_sas_rollout/server"
rollout_resume=(--resume); [[ "$RESUME" == 1 ]] || rollout_resume=(--no-resume)
for ((pass=1; pass<=ROLLOUT_AUTO_RESUME_PASSES; pass++)); do
  set +e
  run_pipeline --phase rollout --data-path "$TRAIN_DATA" --start "$TRAIN_START" --limit "$TRAIN_LIMIT" \
    --num-rollouts "$NUM_ROLLOUTS" --api-base "$(endpoint_pool "$POLICY_PORT")" --api-model sas_policy \
    --temperature "$ROLLOUT_TEMPERATURE" --top-p "$ROLLOUT_TOP_P" --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS" \
    --protocol-retries "$PROTOCOL_RETRIES" --concurrency "$ROLLOUT_CONCURRENCY" --api-timeout "$API_TIMEOUT" \
    --seed "$((GENERATION_SEED + pass - 1))" --source-policy Qwen3-14B-base-strict-SAS \
    --output "$RAW_ROLLOUT" --stats-output "$ROLLOUT_STATS" "${rollout_resume[@]}" "${mock_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/01_sas_rollout/run.log"
  rollout_status=${PIPESTATUS[0]}
  set -e
  [[ "$rollout_status" == 0 ]] && break
  rollout_resume=(--resume)
  echo "[rollout] incomplete pass $pass; retrying missing keys"
done
stop_servers
# Unlike the eval, RL data collection has no denominator to protect: a handful of
# unparseable rollouts can simply be left out. Tolerate a small fraction so a few
# stubborn format failures cannot block training.
"$PYTHON_BIN" - "$ROLLOUT_STATS" "$ROLLOUT_MAX_MISSING_FRACTION" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
expected, completed = int(stats["expected"]), int(stats["completed"])
missing = expected - completed
limit = float(sys.argv[2])
frac = missing / expected if expected else 0.0
print(f"[rollout] missing {missing}/{expected} ({frac:.4%}), tolerance {limit:.4%}")
if frac > limit:
    raise SystemExit(f"[fatal] rollout incomplete beyond tolerance: {missing}/{expected}")
PY

echo "========== 2/3 CONIFER 14B SELF-JUDGE =========="
[[ "$MOCK" == 1 ]] || start_replicas "$MODEL_14B" qwen14b_conifer_sas_judge "$JUDGE_PORT" "$LOG_DIR/02_sas_judge/server"
judge_resume=(--resume); [[ "$RESUME" == 1 ]] || judge_resume=(--no-resume)
for ((pass=1; pass<=JUDGE_AUTO_RESUME_PASSES; pass++)); do
  set +e
  run_pipeline --phase judge --input "$RAW_ROLLOUT" --output "$JUDGED_ROLLOUT" \
    --audit-output "$JUDGE_AUDIT" --stats-output "$JUDGE_STATS" \
    --judge-api-base "$(endpoint_pool "$JUDGE_PORT")" --judge-model qwen14b_conifer_sas_judge \
    --judge-temperature "$JUDGE_TEMPERATURE" --judge-top-p "$JUDGE_TOP_P" \
    --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" --judge-retries "$JUDGE_RETRIES" \
    --outcome-weight "$OUTCOME_WEIGHT" --process-weight "$PROCESS_WEIGHT" \
    --concurrency "$JUDGE_CONCURRENCY" --api-timeout "$API_TIMEOUT" --seed "$GENERATION_SEED" \
    "${judge_resume[@]}" "${mock_args[@]}" 2>&1 | tee -a "$LOG_DIR/02_sas_judge/run.log"
  judge_status=${PIPESTATUS[0]}
  set -e
  [[ "$judge_status" == 0 ]] && break
  judge_resume=(--resume)
  echo "[judge] incomplete pass $pass; retrying missing keys"
done
stop_servers
[[ -s "$JUDGED_ROLLOUT" ]] || fatal "self-judge produced no training data"
judge_failures="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["judge_failures"])' "$JUDGE_STATS")"
[[ "$judge_failures" == 0 ]] || fatal "self-judge remains incomplete after retries: $judge_failures"

if [[ "$MOCK" == 1 ]]; then
  echo "[mock] rollout and judge contracts passed; skipping real 14B training"
  exit 0
fi

echo "========== 3/3 CONIFER 14B FRESH-LORA RWR =========="
if [[ "$RESUME" == 1 && -s "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
  echo "[resume] training already complete: $TRAIN_OUT/final"
  exit 0
fi
train_args=(
  --agent SAS --rollout "$JUDGED_ROLLOUT" --reward-field reward --no-enable-thinking
  --out-dir "$TRAIN_OUT" --model-name-or-path "$MODEL_14B" --kl-coef "$KL_COEF"
  --num-epochs "$NUM_EPOCHS" --learning-rate "$LR" --seed "$GENERATION_SEED"
  --per-device-batch-size "$PER_DEVICE_BATCH" --gradient-accumulation-steps "$GRAD_ACCUM"
  --max-seq-length "$TRAIN_MAX_SEQ" --warmup-ratio "$WARMUP_RATIO"
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
  --logging-steps 10 --save-steps "$TRAIN_SAVE_STEPS" --save-total-limit "$TRAIN_SAVE_TOTAL_LIMIT"
  --report-to tensorboard --gradient-checkpointing --bf16
)
if [[ "$RESUME" == 1 ]] && compgen -G "$TRAIN_OUT/checkpoint-*" >/dev/null; then
  train_args+=(--resume-from-checkpoint latest)
fi
if [[ "$TRAIN_MAX_STEPS" != 0 ]]; then train_args+=(--max-steps "$TRAIN_MAX_STEPS"); fi
launch_args=(--num_processes "$NUM_GPUS" --mixed_precision bf16)
[[ "$NUM_GPUS" == 1 ]] || launch_args+=(--multi_gpu)
env CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$PYTHONPATH_ROOT" \
  "$ACCELERATE" launch "${launch_args[@]}" "$PROJECT_ROOT/scripts/rl_train.py" "${train_args[@]}" \
  2>&1 | tee -a "$LOG_DIR/03_train/run.log"
[[ -s "$TRAIN_OUT/final/adapter_model.safetensors" ]] || fatal "training produced no final adapter"
