#!/usr/bin/env bash
# Qwen3-14B SAS baseline: rollout -> self-judge -> RWR LoRA -> full dev eval.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PIPELINE_PY="${PIPELINE_PY:-/data/conda_envs/qwen35/bin/python}"
VLLM_PY="${VLLM_PY:-/data/conda_envs/deep_research/bin/python}"
VLLM_LIB="${VLLM_LIB:-/data/conda_envs/deep_research/lib}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"

TRAIN_START="${TRAIN_START:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
DEV_START="${DEV_START:-0}"
DEV_LIMIT="${DEV_LIMIT:-2417}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-32}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"

POLICY_PORT="${POLICY_PORT:-8200}"
JUDGE_PORT="${JUDGE_PORT:-8300}"
EVAL_PORT="${EVAL_PORT:-8400}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
RUN_ID="${RUN_ID:-sas14b_self_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/MuSiQue/sas_self_judged_14b_rl/runs/$RUN_ID}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT_ROOT/rl_runs/$RUN_ID}"

RAW="$RUN_DIR/sas_train_raw.jsonl"
JUDGED="$RUN_DIR/sas_train_judged.jsonl"
ADAPTER="$TRAIN_OUT/final"
EVAL_OUT="$RUN_DIR/sas_dev_eval.jsonl"
SERVER_PID=""
SERVER_LOG=""

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" ]] || { echo "[fatal] missing $label: $path" >&2; exit 1; }
}

require_dir() {
  local label="$1" path="$2"
  [[ -d "$path" ]] || { echo "[fatal] missing $label: $path" >&2; exit 1; }
}

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
    echo "[cleanup] vLLM did not stop after 30s; sending KILL"
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
  SERVER_LOG=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server || true
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1" expected_model="$2" started=$SECONDS
  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[fatal] vLLM exited before becoming ready: $SERVER_LOG" >&2
      wait "$SERVER_PID" 2>/dev/null || true
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null)"; then
      if [[ "$response" == *"\"$expected_model\""* ]]; then
        echo "[ready] port=$port model=$expected_model"
        return 0
      fi
    fi
    if (( SECONDS - started >= SERVER_WAIT_TIMEOUT )); then
      echo "[fatal] timed out waiting for port $port model $expected_model" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 5
  done
}

start_policy_server() {
  SERVER_LOG="$RUN_DIR/policy_server.log"
  setsid env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$POLICY_PORT" \
      --model "$MODEL" \
      --served-model-name sas_policy \
      --data-parallel-size 8 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.8 \
      --max-model-len 8192 \
      --max-num-batched-tokens 65536 \
      --max-num-seqs 16 \
      --trust-remote-code \
      --enforce-eager \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$POLICY_PORT" sas_policy
}

start_judge_server() {
  SERVER_LOG="$RUN_DIR/judge_server.log"
  setsid env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$JUDGE_PORT" \
      --model "$MODEL" \
      --served-model-name qwen14b_judge \
      --tensor-parallel-size 8 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.8 \
      --max-model-len 8192 \
      --trust-remote-code \
      --enforce-eager \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$JUDGE_PORT" qwen14b_judge
}

start_eval_server() {
  SERVER_LOG="$RUN_DIR/eval_server.log"
  setsid env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LD_LIBRARY_PATH="${VLLM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$EVAL_PORT" \
      --model "$MODEL" \
      --served-model-name sas_base \
      --data-parallel-size 8 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.8 \
      --max-model-len 8192 \
      --max-num-batched-tokens 65536 \
      --max-num-seqs 16 \
      --trust-remote-code \
      --enforce-eager \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --enable-lora \
      --max-lora-rank 64 \
      --max-loras 1 \
      --max-cpu-loras 1 \
      --lora-dtype auto \
      --lora-modules "sas_policy=$ADAPTER" \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$EVAL_PORT" sas_policy
}

require_file "pipeline Python" "$PIPELINE_PY"
require_file "vLLM Python" "$VLLM_PY"
require_dir "base model" "$MODEL"
require_file "MuSiQue train data" "$PROJECT_ROOT/musique_data/musique_ans_train.jsonl"
require_file "MuSiQue dev data" "$PROJECT_ROOT/musique_data/musique_ans_dev.jsonl"

cat <<EOF
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
TRAIN_OUT=$TRAIN_OUT
TRAIN_RANGE=${TRAIN_START}:$((TRAIN_START + TRAIN_LIMIT))
DEV_RANGE=${DEV_START}:$((DEV_START + DEV_LIMIT))
PIPELINE_PY=$PIPELINE_PY
VLLM_PY=$VLLM_PY
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] paths and configuration validated; no server or training started"
  exit 0
fi

echo "========== 1/4 POLICY ROLLOUT =========="
start_policy_server
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/MuSiQue/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase rollout \
  --split train --start "$TRAIN_START" --limit "$TRAIN_LIMIT" \
  --data-dir musique_data \
  --api-base "http://127.0.0.1:${POLICY_PORT}/v1" \
  --api-model sas_policy \
  --temperature 0.9 --top-p 0.95 --max-new-tokens 1024 \
  --num-rollouts "$NUM_ROLLOUTS" \
  --rollout-concurrency "$ROLLOUT_CONCURRENCY" \
  --output "$RAW" \
  2>&1 | tee "$RUN_DIR/rollout.log"
[[ -s "$RAW" ]] || { echo "[fatal] rollout produced no rows" >&2; exit 1; }
echo "[rollout rows] $(wc -l < "$RAW")"
stop_server

echo "========== 2/4 QWEN14B JUDGE =========="
start_judge_server
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/MuSiQue/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase judge \
  --split train --start "$TRAIN_START" --limit "$TRAIN_LIMIT" \
  --data-dir musique_data \
  --input "$RAW" \
  --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --judge-model qwen14b_judge \
  --alpha 0.6 --judge-retries 2 \
  --judge-concurrency "$JUDGE_CONCURRENCY" \
  --output "$JUDGED" \
  2>&1 | tee "$RUN_DIR/judge.log"
[[ -s "$JUDGED" ]] || { echo "[fatal] judge produced no rows" >&2; exit 1; }
echo "[judged rows] $(wc -l < "$JUDGED")"
stop_server

echo "========== 3/4 RWR LORA TRAIN =========="
ROLLOUT="$JUDGED" \
MODEL="$MODEL" \
OUT_DIR="$TRAIN_OUT" \
NUM_GPUS=8 \
NUM_EPOCHS=2 \
LR=1e-5 \
KL_COEF=0.2 \
PER_DEVICE_BATCH=1 \
GRAD_ACCUM=2 \
MAX_SEQ=4096 \
LORA_RANK=64 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
REPORT_TO=tensorboard \
bash baseline/MuSiQue/sas_self_judged_14b_rl/run_sas_rl_train.sh \
  2>&1 | tee "$RUN_DIR/train.log"
require_file "trained adapter" "$ADAPTER/adapter_model.safetensors"
echo "[adapter] $ADAPTER"

echo "========== 4/4 FULL DEV EVAL =========="
start_eval_server
PYTHONPATH=/data/wangyuheng "$PIPELINE_PY" -u \
  baseline/MuSiQue/sas_self_judged_14b_rl/sas_pipeline.py \
  --phase rollout \
  --split dev --start "$DEV_START" --limit "$DEV_LIMIT" \
  --data-dir musique_data \
  --api-base "http://127.0.0.1:${EVAL_PORT}/v1" \
  --api-model sas_policy \
  --temperature 0.0 --top-p 0.95 --max-new-tokens 1024 \
  --num-rollouts 1 \
  --rollout-concurrency "$ROLLOUT_CONCURRENCY" \
  --output "$EVAL_OUT" \
  2>&1 | tee "$RUN_DIR/eval.log"
[[ -s "$EVAL_OUT" ]] || { echo "[fatal] eval produced no rows" >&2; exit 1; }

"$PIPELINE_PY" - "$EVAL_OUT" "$RUN_DIR/eval_summary.json" "$DEV_LIMIT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected = int(sys.argv[3])
rows = [json.loads(line) for line in input_path.open() if line.strip()]
summary = {
    "rows": len(rows),
    "expected": expected,
    "em": statistics.mean(row["em"] for row in rows) if rows else 0.0,
    "f1": statistics.mean(row["f1"] for row in rows) if rows else 0.0,
    "parser_dropped": expected - len(rows),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
stop_server

echo "========== ALL DONE =========="
echo "Judged rollout: $JUDGED"
echo "Final adapter:  $ADAPTER"
echo "Eval results:   $EVAL_OUT"
echo "Eval summary:   $RUN_DIR/eval_summary.json"
