#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
SCRIPT_DIR="$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl"
PYTHON_COMPAT_DIR="$SCRIPT_DIR/python_compat"
PIPELINE_PY="${PIPELINE_PY:-/data/conda_envs/qwen35/bin/python}"
VLLM_PY="${VLLM_PY:-/data/conda_envs/deep_research/bin/python}"
VLLM_LIB="${VLLM_LIB:-/data/conda_envs/deep_research/lib}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
RUN_ID="${RUN_ID:-gsm_sas14b_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl/runs/$RUN_ID}"
ADAPTER="${ADAPTER:-}"
OUTPUT="${OUTPUT:-$RUN_DIR/sas_dev_eval.jsonl}"
SUMMARY="${SUMMARY:-$RUN_DIR/eval_summary.json}"
PORT="${PORT:-8400}"
START="${START:-0}"
LIMIT="${LIMIT:-132}"
CONCURRENCY="${CONCURRENCY:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
GENERATION_SEED="${GENERATION_SEED:-42}"
API_TIMEOUT="${API_TIMEOUT:-900}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_LOG="$RUN_DIR/eval_server.log"
SERVER_PID=""

cd "$PROJECT_ROOT"
[[ -s "$DATA_PATH" ]] || { echo "[fatal] missing data: $DATA_PATH" >&2; exit 1; }
[[ -s "$MODEL/config.json" ]] || { echo "[fatal] missing model: $MODEL" >&2; exit 1; }
[[ -x "$PIPELINE_PY" && -x "$VLLM_PY" ]] || { echo "[fatal] required Python executable missing" >&2; exit 1; }
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "[fatal] invalid GPU_IDS: $GPU_IDS" >&2; exit 1; }
[[ "$DP_SIZE" =~ ^[1-9][0-9]*$ && "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fatal] DP_SIZE and TP_SIZE must be positive integers" >&2
  exit 1
}
GPU_COUNT="$(awk -F',' '{print NF}' <<<"$GPU_IDS")"
(( DP_SIZE * TP_SIZE == GPU_COUNT )) || {
  echo "[fatal] DP_SIZE*TP_SIZE must equal the number of GPU_IDS ($GPU_COUNT)" >&2
  exit 1
}
[[ "$ENABLE_THINKING" == "0" || "$ENABLE_THINKING" == "1" ]] || {
  echo "[fatal] ENABLE_THINKING must be 0 or 1" >&2
  exit 1
}
mkdir -p "$RUN_DIR"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

server_args=(
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1 --port "$PORT"
  --model "$MODEL" --served-model-name sas_policy
  --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS"
  --trust-remote-code --enforce-eager
  --enable-prefix-caching --enable-chunked-prefill
)
if [[ -n "$ADAPTER" ]]; then
  server_args=(
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1 --port "$PORT"
    --model "$MODEL" --served-model-name sas_base
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code --enforce-eager
    --enable-prefix-caching --enable-chunked-prefill
  )
  server_args+=(
    --enable-lora --max-lora-rank 64 --max-loras 1 --max-cpu-loras 1
    --lora-dtype auto --lora-modules "sas_policy=$ADAPTER"
  )
fi

echo "RUN_ID=$RUN_ID"
echo "MODEL=$MODEL"
echo "ADAPTER=${ADAPTER:-BASE}"
echo "OUTPUT=$OUTPUT"
echo "TOPOLOGY=tp${TP_SIZE}xdp${DP_SIZE} gpus=$GPU_IDS"
echo "ENABLE_THINKING=$ENABLE_THINKING"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] eval paths and configuration validated"
  exit 0
fi

if [[ -n "$ADAPTER" ]]; then
  [[ -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "[fatal] missing adapter: $ADAPTER" >&2; exit 1; }
fi

setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${server_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

started=$SECONDS
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[fatal] eval server exited before ready" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if response="$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" && [[ "$response" == *'"sas_policy"'* ]]; then
    echo "[ready] eval server"
    break
  fi
  if (( SECONDS - started >= SERVER_WAIT_TIMEOUT )); then
    echo "[fatal] timed out waiting for eval server" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

thinking_arg="--enable-thinking"
[[ "$ENABLE_THINKING" == "1" ]] || thinking_arg="--no-enable-thinking"
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/GSM-Hard/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase rollout --data-path "$DATA_PATH" --start "$START" --limit "$LIMIT" \
  --api-base "http://127.0.0.1:${PORT}/v1" --api-model sas_policy \
  --api-timeout "$API_TIMEOUT" --temperature "$TEMPERATURE" --top-p "$TOP_P" \
  --max-new-tokens "$MAX_NEW_TOKENS" "$thinking_arg" \
  --seed "$GENERATION_SEED" --num-rollouts 1 --concurrency "$CONCURRENCY" --include-failures \
  --output "$OUTPUT" --audit-output "$RUN_DIR/eval_audit.jsonl" \
  --stats-output "$SUMMARY" \
  2>&1 | tee "$RUN_DIR/eval.log"

[[ -s "$OUTPUT" ]] || { echo "[fatal] eval produced no rows" >&2; exit 1; }
echo "[done] raw eval=$OUTPUT"
echo "[done] summary=$SUMMARY"
