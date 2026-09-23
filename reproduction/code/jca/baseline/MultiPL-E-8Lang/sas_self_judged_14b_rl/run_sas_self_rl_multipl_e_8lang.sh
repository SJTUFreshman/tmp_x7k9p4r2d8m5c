#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/wangyuheng/models/Qwen3-14B}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Code/multipl_e_8lang_benchmark/splits/unified_train70_test30_seed7658190907657085414}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_multipl_e_8lang_sas14b_self_rl}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/logs/multipl_e_8lang_baselines/sas_self_judged_14b_rl/$RUN_ID}"
POLICY_PORT="${POLICY_PORT:-8401}"
JUDGE_PORT="${JUDGE_PORT:-8402}"
ALL_GPUS="${ALL_GPUS:-0,1,2,3,4,5,6,7}"
POLICY_GPUS="${POLICY_GPUS:-$ALL_GPUS}"
JUDGE_GPUS="${JUDGE_GPUS:-$ALL_GPUS}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EVAL_IMAGE="${EVAL_IMAGE:-multipl-e-evaluation:jca-current}"
DOCKER_EXEC="${DOCKER_EXEC:-docker}"
EVAL_SHARDS="${EVAL_SHARDS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SMOKE_MODE="${SMOKE_MODE:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
REUSE_TRAIN_ROLLOUT="${REUSE_TRAIN_ROLLOUT:-}"
POLICY_API="http://127.0.0.1:$POLICY_PORT/v1"
JUDGE_API="http://127.0.0.1:$JUDGE_PORT/v1"

PIPELINE="$PROJECT_ROOT/Code/MultiPL-E/scripts/evaluate_multipl_e_parallel.py"
SAS_PIPELINE="$SCRIPT_DIR/sas_pipeline.py"
PREPARE="$SCRIPT_DIR/prepare_sas_completions.py"
OUTCOMES="$SCRIPT_DIR/build_sas_outcomes.py"
VALIDATE_ROLLOUT="$SCRIPT_DIR/validate_sas_rollout.py"
TRAIN_INPUT="$DATA_PATH/train.jsonl"
TEST_INPUT="$DATA_PATH/test.jsonl"
TRAIN_RECORDS="$RUN_DIR/train_rollout.jsonl"
TRAIN_SCORED="$RUN_DIR/train_scored.jsonl"
TRAIN_REJECTED="$RUN_DIR/train_rejected.jsonl"
TRAIN_COMPLETIONS="$RUN_DIR/train_completions"
BASE_TEST_RECORDS="$RUN_DIR/base_test_rollout.jsonl"
BASE_TEST_COMPLETIONS="$RUN_DIR/base_test_completions"
TEST_RECORDS="$RUN_DIR/test_rollout.jsonl"
TEST_COMPLETIONS="$RUN_DIR/test_completions"
OUTCOME_PATH="$RUN_DIR/train_outcomes.jsonl"
RL_OUT="$RUN_DIR/rl_train"

mkdir -p "$RUN_DIR"
if [[ "$SMOKE_MODE" == "1" ]]; then
  TRAIN_LIMIT="${TRAIN_LIMIT:-2}"
  TEST_LIMIT="${TEST_LIMIT:-2}"
  EVAL_SHARDS="${SMOKE_EVAL_SHARDS:-2}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
else
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
fi

cat >"$RUN_DIR/config.env" <<EOF
PIPELINE_VERSION=normalized_base_control_v2
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
MODEL_PATH=$MODEL_PATH
DATA_PATH=$DATA_PATH
POLICY_GPUS=$POLICY_GPUS
JUDGE_GPUS=$JUDGE_GPUS
ALL_GPUS=$ALL_GPUS
TP_SIZE=$TP_SIZE
MAX_MODEL_LEN=$MAX_MODEL_LEN
EVAL_IMAGE=$EVAL_IMAGE
EVAL_SHARDS=$EVAL_SHARDS
SMOKE_MODE=$SMOKE_MODE
TRAIN_LIMIT=$TRAIN_LIMIT
TEST_LIMIT=$TEST_LIMIT
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS
RESUME=$RESUME
REUSE_TRAIN_ROLLOUT=$REUSE_TRAIN_ROLLOUT
EOF
printf '%q ' "$0" "$@" >"$RUN_DIR/command.txt"
printf '\n' >>"$RUN_DIR/command.txt"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry run configuration written to $RUN_DIR"
  exit 0
fi

if [[ "$RESUME" == "1" ]]; then
  exec > >(tee -a "$RUN_DIR/run.log") 2>&1
else
  exec > >(tee "$RUN_DIR/run.log") 2>&1
fi
echo "running" >"$RUN_DIR/status.txt"
SERVER_PIDS=()
LAST_SERVER_PID=""
CURRENT_STAGE="initializing"
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_servers
  if [[ "$status" -eq 0 ]]; then
    echo "completed" >"$RUN_DIR/status.txt"
  else
    printf 'failed status=%s stage=%s\n' "$status" "$CURRENT_STAGE" >"$RUN_DIR/status.txt"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

start_server() {
  local log_name="$1" served_name="$2" port="$3" gpus="$4"; shift 4
  setsid env CUDA_VISIBLE_DEVICES="$gpus" \
    PYTHONPATH="$PROJECT_ROOT/baseline/MuSiQue/vllm_compat${PYTHONPATH:+:$PYTHONPATH}" \
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "$port" --model "$MODEL_PATH" \
    --served-model-name "$served_name" --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 --gpu-memory-utilization 0.80 \
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code "$@" \
    >"$RUN_DIR/${log_name}_server.log" 2>&1 &
  SERVER_PIDS+=("$!")
  LAST_SERVER_PID="$!"
}

start_policy_server() {
  start_server sas_policy sas_policy "$POLICY_PORT" "$POLICY_GPUS" "$@"
  POLICY_PID="$LAST_SERVER_PID"
  wait_server "$POLICY_PORT" sas_policy "$POLICY_PID"
}

start_judge_server() {
  start_server qwen14b_judge qwen14b_judge "$JUDGE_PORT" "$JUDGE_GPUS" "$@"
  JUDGE_PID="$LAST_SERVER_PID"
  wait_server "$JUDGE_PORT" qwen14b_judge "$JUDGE_PID"
}

stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]}"; do
    kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "$pid" >/dev/null 2>&1 || true
  done
  SERVER_PIDS=()
}

wait_server() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json, os, sys, time, urllib.request
port, expected, pid = sys.argv[1:]
deadline = time.time() + 900
while time.time() < deadline:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        raise SystemExit(f"server process {pid} for {expected} exited before becoming ready")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            if expected in [item.get("id") for item in json.load(response).get("data", [])]: raise SystemExit(0)
    except Exception: time.sleep(3)
raise SystemExit(f"server {expected} did not become ready")
PY
}

evaluate_dir() {
  local input="$1"
  "$PYTHON_BIN" "$PIPELINE" --input-dir "$input" --output-dir "$input" \
    --image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC" --shards "$EVAL_SHARDS" \
    --inner-workers 1 --quiet
}

[[ -f "$TRAIN_INPUT" && -f "$TEST_INPUT" ]] || { echo "missing unified train/test jsonl" >&2; exit 2; }
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || { echo "missing base model: $MODEL_PATH" >&2; exit 2; }

if [[ -n "$REUSE_TRAIN_ROLLOUT" ]]; then
  [[ -f "$REUSE_TRAIN_ROLLOUT" ]] || { echo "missing reusable train rollout: $REUSE_TRAIN_ROLLOUT" >&2; exit 2; }
  [[ "$(readlink -f "$REUSE_TRAIN_ROLLOUT")" != "$(readlink -m "$TRAIN_RECORDS")" ]] || {
    echo "REUSE_TRAIN_ROLLOUT must come from a different run directory" >&2
    exit 2
  }
  cp --reflink=auto "$REUSE_TRAIN_ROLLOUT" "$TRAIN_RECORDS"
  EXPECTED_TRAIN_COUNT="${TRAIN_LIMIT:-$(wc -l < "$TRAIN_INPUT")}"
  "$PYTHON_BIN" "$VALIDATE_ROLLOUT" --input "$TRAIN_INPUT" --rollout "$TRAIN_RECORDS" \
    --expected-count "$EXPECTED_TRAIN_COUNT"
fi

echo "[1/7] start base policy and judge servers"
CURRENT_STAGE="starting base policy server"
start_policy_server

echo "[2/7] train70 policy rollout"
CURRENT_STAGE="train policy rollout"
if [[ -n "$REUSE_TRAIN_ROLLOUT" ]]; then
  echo "reused_train_rollout=$REUSE_TRAIN_ROLLOUT"
else
  LIMIT_ARGS=()
  if [[ -n "$TRAIN_LIMIT" ]]; then LIMIT_ARGS+=(--limit "$TRAIN_LIMIT"); fi
  if [[ "$RESUME" == "1" ]]; then LIMIT_ARGS+=(--resume); fi
  "$PYTHON_BIN" "$SAS_PIPELINE" --input "$TRAIN_INPUT" --output "$TRAIN_RECORDS" \
    --policy-api "$POLICY_API" --temperature 0.7 --max-new-tokens 4096 \
    --max-concurrency "$MAX_CONCURRENCY" --policy-only "${LIMIT_ARGS[@]}"
fi

echo "[3/7] normalize, execute, and judge train70"
CURRENT_STAGE="train execution and judge scoring"
stop_servers
"$PYTHON_BIN" "$PREPARE" --rollout "$TRAIN_RECORDS" --output-dir "$TRAIN_COMPLETIONS"
evaluate_dir "$TRAIN_COMPLETIONS"
"$PYTHON_BIN" "$OUTCOMES" --completions-dir "$TRAIN_COMPLETIONS" --output "$OUTCOME_PATH"
start_judge_server
"$PYTHON_BIN" "$SCRIPT_DIR/score_sas_rollouts.py" --rollout "$TRAIN_RECORDS" --outcomes "$OUTCOME_PATH" \
  --output "$TRAIN_SCORED" --rejected-output "$TRAIN_REJECTED" \
  --judge-api "$JUDGE_API" --judge-model qwen14b_judge --max-concurrency "$MAX_CONCURRENCY"

echo "[4/7] evaluate same-protocol Qwen3-14B Base"
CURRENT_STAGE="base test30 rollout and execution"
stop_servers
start_policy_server
LIMIT_ARGS=()
if [[ -n "$TEST_LIMIT" ]]; then LIMIT_ARGS+=(--limit "$TEST_LIMIT"); fi
"$PYTHON_BIN" "$SAS_PIPELINE" --input "$TEST_INPUT" --output "$BASE_TEST_RECORDS" \
  --policy-api "$POLICY_API" --policy-model sas_policy --temperature 0.0 \
  --max-new-tokens 4096 --max-concurrency "$MAX_CONCURRENCY" --policy-only "${LIMIT_ARGS[@]}"
"$PYTHON_BIN" "$PREPARE" --rollout "$BASE_TEST_RECORDS" --output-dir "$BASE_TEST_COMPLETIONS"
evaluate_dir "$BASE_TEST_COMPLETIONS"
"$PYTHON_BIN" "$OUTCOMES" --completions-dir "$BASE_TEST_COMPLETIONS" --output "$RUN_DIR/base_test_outcomes.jsonl"
SUMMARY_ARGS=()
if [[ -n "$TEST_LIMIT" ]]; then SUMMARY_ARGS+=(--allow-partial); fi
"$PYTHON_BIN" "$PROJECT_ROOT/Code/MultiPL-E/scripts/summarize_multipl_e_sas.py" \
  --split-manifest "$PROJECT_ROOT/Code/multipl_e_8lang_benchmark/splits/humaneval_train70_test30_seed7658190907657085414/split_manifest.json" \
  "$PROJECT_ROOT/Code/multipl_e_8lang_benchmark/splits/mbpp_train70_test30_seed7658190907657085414/split_manifest.json" \
  --completions-dir "$BASE_TEST_COMPLETIONS" --output "$RUN_DIR/base_test_scores.json" \
  --text-output "$RUN_DIR/base_summary.txt" --model-name "SAS-Qwen3-14B-Base" \
  --evaluation "sas-base-test30" "${SUMMARY_ARGS[@]}"

echo "[5/7] train fresh SAS LoRA"
CURRENT_STAGE="fresh LoRA training"
stop_servers
OUT_DIR="$RL_OUT" "$ACCELERATE" launch --num_processes "${NUM_GPUS:-8}" --mixed_precision bf16 \
  "$PROJECT_ROOT/scripts/rl_train.py" --agent SAS --rollout "$TRAIN_SCORED" --reward-field reward \
  --model-name-or-path "$MODEL_PATH" --out-dir "$RL_OUT" --kl-coef 0.05 --num-epochs 1 \
  --learning-rate 3e-6 --per-device-batch-size "$TRAIN_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
  --max-seq-length 12000 --lora-rank 32 --lora-alpha 32 --lora-dropout 0.05 --bf16 --gradient-checkpointing

echo "[6/7] test30 rollout with trained adapter"
CURRENT_STAGE="trained policy test rollout and execution"
ADAPTER="$RL_OUT/final"
[[ -f "$ADAPTER/adapter_config.json" ]] || { echo "missing trained adapter: $ADAPTER" >&2; exit 2; }
start_server sas_policy_trained sas_base "$POLICY_PORT" "$POLICY_GPUS" --enable-lora --max-lora-rank 32 --lora-modules "sas_policy=$ADAPTER"
POLICY_PID="$LAST_SERVER_PID"
wait_server "$POLICY_PORT" sas_policy "$POLICY_PID"
LIMIT_ARGS=()
if [[ -n "$TEST_LIMIT" ]]; then LIMIT_ARGS+=(--limit "$TEST_LIMIT"); fi
if [[ "$RESUME" == "1" ]]; then LIMIT_ARGS+=(--resume); fi
"$PYTHON_BIN" "$SAS_PIPELINE" --input "$TEST_INPUT" --output "$TEST_RECORDS" \
  --policy-api "$POLICY_API" --policy-model sas_policy --temperature 0.0 \
  --max-new-tokens 4096 --max-concurrency "$MAX_CONCURRENCY" --policy-only "${LIMIT_ARGS[@]}"
"$PYTHON_BIN" "$PREPARE" --rollout "$TEST_RECORDS" --output-dir "$TEST_COMPLETIONS"
evaluate_dir "$TEST_COMPLETIONS"
"$PYTHON_BIN" "$OUTCOMES" --completions-dir "$TEST_COMPLETIONS" --output "$RUN_DIR/test_outcomes.jsonl"

echo "[7/7] summarize trained test30"
CURRENT_STAGE="test summary"
SUMMARY_ARGS=()
if [[ -n "$TEST_LIMIT" ]]; then SUMMARY_ARGS+=(--allow-partial); fi
"$PYTHON_BIN" "$PROJECT_ROOT/Code/MultiPL-E/scripts/summarize_multipl_e_sas.py" \
  --split-manifest "$PROJECT_ROOT/Code/multipl_e_8lang_benchmark/splits/humaneval_train70_test30_seed7658190907657085414/split_manifest.json" \
  "$PROJECT_ROOT/Code/multipl_e_8lang_benchmark/splits/mbpp_train70_test30_seed7658190907657085414/split_manifest.json" \
  --completions-dir "$TEST_COMPLETIONS" --output "$RUN_DIR/test_scores.json" \
  --text-output "$RUN_DIR/summary.txt" --model-name "SAS-Qwen3-14B-RL" \
  --evaluation "sas-self-rl-test30" "${SUMMARY_ARGS[@]}"
echo "completed: $RUN_DIR"
