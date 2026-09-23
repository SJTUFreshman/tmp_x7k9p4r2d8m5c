#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-${VLLM_PYTHON_BIN%/bin/python}/lib}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Math/data/MATH}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-8B}"
SPLIT="${SPLIT:-test}"; START="${START:-0}"; LIMIT="${LIMIT:-5000}"
PROBLEM_IDS_FILE="${PROBLEM_IDS_FILE:-}"
API_MODEL="${API_MODEL:-single_agent}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8501}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DP_SIZE="${DP_SIZE:-8}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"; MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
TEMPERATURE="${TEMPERATURE:-0.0}"; TOP_P="${TOP_P:-0.95}"; SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"; REQUIRE_THINKING="${REQUIRE_THINKING:-0}"
STEP_RETRIES="${STEP_RETRIES:-1}"
LENGTH_RETRIES="${LENGTH_RETRIES:-1}"
RUN_ID="${RUN_ID:-math_single_${MODEL##*/}_${SPLIT}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/logs/math_baselines/single_agent/${RUN_ID}}"
OUTPUT="${OUTPUT:-${RUN_DIR}/results.jsonl}"; SUMMARY="${SUMMARY:-${RUN_DIR}/summary.json}"
SERVER_LOG="${SERVER_LOG:-${RUN_DIR}/server.log}"; RUN_LOG="${RUN_LOG:-${RUN_DIR}/run.log}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"; STOP_TIMEOUT="${STOP_TIMEOUT:-45}"
mkdir -p "$RUN_DIR"
[[ "$SPLIT" == test || "${ALLOW_TRAIN_SPLIT:-0}" == 1 ]] || { echo "SPLIT must be test" >&2; exit 2; }
[[ "$ENABLE_THINKING" =~ ^[01]$ && "$REQUIRE_THINKING" =~ ^[01]$ ]] || { echo "thinking flags must be 0 or 1" >&2; exit 2; }
[[ "$REQUIRE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || { echo "REQUIRE_THINKING=1 requires ENABLE_THINKING=1" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" && -s "$MODEL/config.json" ]] || { echo "missing Python or model: $MODEL" >&2; exit 2; }
[[ "$DP_SIZE" == 8 && "$GPU_IDS" == 0,1,2,3,4,5,6,7 ]] || { echo "default launcher requires all 8 GPUs and DP=8" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'model=%s GPUs=%s DP=%s concurrency=%s split=%s start=%s limit=%s ids=%s thinking=%s/%s temperature=%s top_p=%s max_new_tokens=%s max_model_len=%s\n' "$MODEL" "$GPU_IDS" "$DP_SIZE" "$MAX_CONCURRENCY" "$SPLIT" "$START" "$LIMIT" "${PROBLEM_IDS_FILE:-contiguous}" "$ENABLE_THINKING" "$REQUIRE_THINKING" "$TEMPERATURE" "$TOP_P" "$MAX_NEW_TOKENS" "$MAX_MODEL_LEN"
  exit 0
fi

cat >"${RUN_DIR}/config.env" <<EOF
RUN_ID=$RUN_ID
MODEL=$MODEL
SPLIT=$SPLIT
START=$START
LIMIT=$LIMIT
PROBLEM_IDS_FILE=$PROBLEM_IDS_FILE
GPU_IDS=$GPU_IDS
TP_SIZE=1
DP_SIZE=$DP_SIZE
MAX_CONCURRENCY=$MAX_CONCURRENCY
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
ENABLE_THINKING=$ENABLE_THINKING
REQUIRE_THINKING=$REQUIRE_THINKING
TEMPERATURE=$TEMPERATURE
TOP_P=$TOP_P
GENERATION_SEED=$SEED
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
MAX_MODEL_LEN=$MAX_MODEL_LEN
STEP_RETRIES=$STEP_RETRIES
LENGTH_RETRIES=$LENGTH_RETRIES
EOF

SERVER_PID=""
cleanup() { local code=$?; trap - EXIT INT TERM; if [[ -n "$SERVER_PID" ]]; then kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true; fi; exit "$code"; }
trap cleanup EXIT INT TERM
wait_ready() {
  "$PYTHON_BIN" - "$HOST" "$PORT" "$API_MODEL" "$SERVER_WAIT_TIMEOUT" "$SERVER_PID" <<'PY'
import json,os,sys,time,urllib.request
host,port,model,timeout,server_pid=sys.argv[1:]; end=time.time()+float(timeout)
while time.time()<end:
 try:
  os.kill(int(server_pid),0)
 except OSError:
  print(f"server exited before ready (pid={server_pid})",file=sys.stderr)
  raise SystemExit(1)
 try:
  with urllib.request.urlopen(f'http://{host}:{port}/v1/models',timeout=5) as r: names=[x['id'] for x in json.load(r)['data']]
  if model in names: print('server ready',names); raise SystemExit(0)
 except Exception: pass
 time.sleep(3)
raise SystemExit('server timeout')
PY
}

echo "starting single-agent model=$MODEL GPUs=$GPU_IDS DP=$DP_SIZE concurrency=$MAX_CONCURRENCY" | tee "$RUN_LOG"
setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" --port "$PORT" --model "$MODEL" --served-model-name "$API_MODEL" \
  --tensor-parallel-size 1 --data-parallel-size "$DP_SIZE" --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code --enforce-eager --enable-prefix-caching --enable-chunked-prefill >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
wait_ready
thinking_args=(--no-enable-thinking --no-require-thinking)
[[ "$ENABLE_THINKING" == 1 ]] && thinking_args[0]=--enable-thinking
[[ "$REQUIRE_THINKING" == 1 ]] && thinking_args[1]=--require-thinking
problem_ids_args=()
[[ -z "$PROBLEM_IDS_FILE" ]] || problem_ids_args=(--problem-ids-file "$PROBLEM_IDS_FILE")
PYTHONPATH="$PROJECT_ROOT/.." "$PYTHON_BIN" -u "$SCRIPT_DIR/run_math_single_agent.py" \
  --data-root "$DATA_ROOT" --split "$SPLIT" --start "$START" --limit "$LIMIT" \
  "${problem_ids_args[@]}" \
  --api-base "http://${HOST}:${PORT}/v1" --api-model "$API_MODEL" --model "$MODEL" \
  --output "$OUTPUT" --summary "$SUMMARY" \
  --max-concurrency "$MAX_CONCURRENCY" --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --seed "$SEED" \
  --retries "$STEP_RETRIES" \
  --length-retries "$LENGTH_RETRIES" \
  "${thinking_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"
echo "done: $SUMMARY"
