#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
SCRIPT_DIR="$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl"
PYTHON_COMPAT_DIR="$SCRIPT_DIR/python_compat"
PIPELINE_PY="${PIPELINE_PY:-/data/conda_envs/qwen35/bin/python}"
VLLM_PY="${VLLM_PY:-/data/conda_envs/deep_research/bin/python}"
VLLM_LIB="${VLLM_LIB:-/data/conda_envs/deep_research/lib}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl}"
RUN_ID="${RUN_ID:-gsm_sas14b_self_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl/runs/$RUN_ID}"
RAW="${RAW:-$RUN_DIR/sas_train_raw.jsonl}"
JUDGED="${JUDGED:-$RUN_DIR/sas_train_judged.jsonl}"

START="${START:-0}"
LIMIT="${LIMIT:-1187}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-32}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
POLICY_PORT="${POLICY_PORT:-8200}"
JUDGE_PORT="${JUDGE_PORT:-8300}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_PID=""
SERVER_LOG=""

cd "$PROJECT_ROOT"
[[ -s "$DATA_PATH" ]] || { echo "[fatal] missing data: $DATA_PATH" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "[fatal] missing model: $MODEL" >&2; exit 1; }
mkdir -p "$RUN_DIR"

stop_server() {
  local pid="${SERVER_PID:-}"
  [[ -n "$pid" ]] || return 0
  echo "[cleanup] stopping vLLM process group $pid"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server || true
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1" model_name="$2" started=$SECONDS
  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[fatal] vLLM exited before ready: $SERVER_LOG" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" && [[ "$response" == *"\"$model_name\""* ]]; then
      echo "[ready] port=$port model=$model_name"
      return 0
    fi
    if (( SECONDS - started >= SERVER_WAIT_TIMEOUT )); then
      echo "[fatal] timed out waiting for $model_name" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 5
  done
}

start_policy_server() {
  SERVER_LOG="$RUN_DIR/policy_server.log"
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port "$POLICY_PORT" \
      --model "$MODEL" --served-model-name sas_policy \
      --data-parallel-size 8 --dtype bfloat16 \
      --gpu-memory-utilization 0.8 --max-model-len 8192 \
      --max-num-batched-tokens 65536 --max-num-seqs 32 \
      --trust-remote-code --enforce-eager \
      --enable-prefix-caching --enable-chunked-prefill \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$POLICY_PORT" sas_policy
}

start_judge_server() {
  SERVER_LOG="$RUN_DIR/judge_server.log"
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port "$JUDGE_PORT" \
      --model "$MODEL" --served-model-name qwen14b_judge \
      --tensor-parallel-size 8 --dtype bfloat16 \
      --gpu-memory-utilization 0.8 --max-model-len 16384 \
      --trust-remote-code --enforce-eager \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$JUDGE_PORT" qwen14b_judge
}

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "train_range=${START}:$((START + LIMIT)) rollouts_per_problem=$NUM_ROLLOUTS"
echo "RAW=$RAW"
echo "JUDGED=$JUDGED"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] data-generation paths and configuration validated"
  exit 0
fi

echo "========== 1/2 POLICY ROLLOUT =========="
start_policy_server
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/GSM-Hard/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase rollout --data-path "$DATA_PATH" --start "$START" --limit "$LIMIT" \
  --api-base "http://127.0.0.1:${POLICY_PORT}/v1" --api-model sas_policy \
  --temperature 0.9 --top-p 0.95 --max-new-tokens 1024 --no-enable-thinking \
  --seed 42 --num-rollouts "$NUM_ROLLOUTS" --concurrency "$ROLLOUT_CONCURRENCY" \
  --output "$RAW" --audit-output "$RUN_DIR/rollout_audit.jsonl" \
  --stats-output "$RUN_DIR/rollout_stats.json" \
  2>&1 | tee "$RUN_DIR/rollout.log"
[[ -s "$RAW" ]] || { echo "[fatal] rollout produced no valid rows" >&2; exit 1; }
stop_server

echo "========== 2/2 QWEN14B PROCESS JUDGE =========="
start_judge_server
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/GSM-Hard/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase judge --data-path "$DATA_PATH" --start "$START" --limit "$LIMIT" \
  --input "$RAW" --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --judge-model qwen14b_judge --judge-temperature 0.0 --judge-top-p 0.95 \
  --judge-max-new-tokens 8192 --judge-enable-thinking --judge-retries 2 \
  --outcome-weight 0.6 --process-weight 0.4 --concurrency "$JUDGE_CONCURRENCY" \
  --output "$JUDGED" --audit-output "$RUN_DIR/judge_audit.jsonl" \
  --stats-output "$RUN_DIR/judge_stats.json" \
  2>&1 | tee "$RUN_DIR/judge.log"
[[ -s "$JUDGED" ]] || { echo "[fatal] judge produced no valid rows" >&2; exit 1; }
stop_server

echo "[done] raw_rows=$(wc -l < "$RAW") judged_rows=$(wc -l < "$JUDGED")"
