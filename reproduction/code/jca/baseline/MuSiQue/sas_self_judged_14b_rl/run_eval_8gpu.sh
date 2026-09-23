#!/usr/bin/env bash
# Evaluate one trained Qwen3-14B SAS LoRA on MuSiQue dev with DP=8.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PIPELINE_PY="${PIPELINE_PY:-/data/conda_envs/qwen35/bin/python}"
VLLM_PY="${VLLM_PY:-/data/conda_envs/deep_research/bin/python}"
VLLM_LIB="${VLLM_LIB:-/data/conda_envs/deep_research/lib}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"
RUN_ID="${RUN_ID:-sas14b_self_rl_20260817_161921}"
DEFAULT_ADAPTER="$PROJECT_ROOT/rl_runs/$RUN_ID/final"
ADAPTER="${ADAPTER-$DEFAULT_ADAPTER}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/MuSiQue/sas_self_judged_14b_rl/runs/$RUN_ID}"
OUTPUT="${OUTPUT:-$RUN_DIR/sas_dev_eval.jsonl}"
SUMMARY="${SUMMARY:-$RUN_DIR/eval_summary.json}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/musique_data}"
PORT="${PORT:-8400}"
START="${START:-0}"
LIMIT="${LIMIT:-2417}"
CONCURRENCY="${CONCURRENCY:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_LOG="$RUN_DIR/eval_server.log"
SERVER_PID=""

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"
[[ -s "$MODEL/config.json" ]] || { echo "[fatal] missing model: $MODEL" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "[fatal] missing data directory: $DATA_DIR" >&2; exit 1; }
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
if [[ -n "$ADAPTER" && ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "[fatal] missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi

echo "RUN_ID=$RUN_ID"
echo "MODEL=$MODEL"
echo "ADAPTER=${ADAPTER:-BASE}"
echo "OUTPUT=$OUTPUT"
echo "TOPOLOGY=tp${TP_SIZE}xdp${DP_SIZE} gpus=$GPU_IDS"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] eval paths and configuration validated"
  exit 0
fi

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

SERVER_MODEL_NAME="sas_policy"
[[ -z "$ADAPTER" ]] || SERVER_MODEL_NAME="sas_base"
server_args=(
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1
  --port "$PORT"
  --model "$MODEL"
  --served-model-name "$SERVER_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --trust-remote-code
  --enforce-eager
  --enable-prefix-caching
  --enable-chunked-prefill
)
if [[ -n "$ADAPTER" ]]; then
  server_args+=(
    --enable-lora
    --max-lora-rank 64
    --max-loras 1
    --max-cpu-loras 1
    --lora-dtype auto
    --lora-modules "sas_policy=$ADAPTER"
  )
fi

setsid env \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${server_args[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

started=$SECONDS
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[fatal] eval vLLM exited before becoming ready" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if response="$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)"; then
    if [[ "$response" == *'"sas_policy"'* ]]; then
      echo "[ready] eval server exposes sas_policy"
      break
    fi
  fi
  if (( SECONDS - started >= SERVER_WAIT_TIMEOUT )); then
    echo "[fatal] timed out waiting for eval server" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/MuSiQue/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase rollout \
  --split dev --start "$START" --limit "$LIMIT" \
  --data-dir "$DATA_DIR" \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --api-model sas_policy \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-rollouts 1 \
  --rollout-concurrency "$CONCURRENCY" \
  --output "$OUTPUT" \
  2>&1 | tee "$RUN_DIR/eval.log"

[[ -s "$OUTPUT" ]] || { echo "[fatal] eval produced no rows" >&2; exit 1; }
"$PIPELINE_PY" - "$OUTPUT" "$SUMMARY" "$LIMIT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

source, destination, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in source.open() if line.strip()]
summary = {
    "rows": len(rows),
    "expected": expected,
    "em": statistics.mean(row["em"] for row in rows) if rows else 0.0,
    "f1": statistics.mean(row["f1"] for row in rows) if rows else 0.0,
    "parser_dropped": expected - len(rows),
}
destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

stop_server
echo "[done] raw eval: $OUTPUT"
echo "[done] summary:  $SUMMARY"
