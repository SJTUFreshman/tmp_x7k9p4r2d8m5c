#!/usr/bin/env bash
# Conifer AFlow workflow search on 8 GPUs (2+2+2+2).
#
# Starts A1/A2/A3 plus the Qwen3-14B optimizer, runs the 20x20 MCTS search on a
# seeded train-20 subset, then stops every server.  The reported test split is
# never read here; evaluation of the selected workflow is a separate stage that
# reuses Conifer's normal eval path with --baseline-aflow-workflow-file.
#
# See conifer_training_hub/AFLOW_SEARCH_MIGRATION.md.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HUB_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${HUB_ROOT}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/drb_py311_clean/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/drb_py311_clean/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"

DATA_PATH="${DATA_PATH:-${HUB_ROOT}/01_dataset/processed/train.jsonl}"
INITIAL_WORKFLOW="${INITIAL_WORKFLOW:-${SCRIPT_DIR}/workflows/round_00_initial.py}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_OPT="${MODEL_OPT:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_OPT="${SERVED_OPT:-Optimizer_14B}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8411}"; A2_PORT="${A2_PORT:-8412}"
A3_PORT="${A3_PORT:-8413}"; OPT_PORT="${OPT_PORT:-8414}"
A1_GPUS="${A1_GPUS:-0,1}"; A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5}"; OPT_GPUS="${OPT_GPUS:-6,7}"
A1_TP="${A1_TP:-2}"; A2_TP="${A2_TP:-2}"; A3_TP="${A3_TP:-2}"; OPT_TP="${OPT_TP:-2}"

# Conifer-internal budgets: exec 1024 matches PROTOCOL_MAX_NEW_TOKENS used by
# every other Conifer arm; opt 2048 matches AFlow on the other four datasets.
MAX_NEW_TOKENS_EXEC="${MAX_NEW_TOKENS_EXEC:-1024}"
MAX_NEW_TOKENS_OPT="${MAX_NEW_TOKENS_OPT:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TEMPERATURE_EXEC="${TEMPERATURE_EXEC:-0.7}"
TEMPERATURE_OPT="${TEMPERATURE_OPT:-0.8}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
SEARCH_SIZE="${SEARCH_SIZE:-20}"
SEARCH_SEED="${SEARCH_SEED:-20260810}"
RNG_SEED="${RNG_SEED:-20260810}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
API_TIMEOUT="${API_TIMEOUT:-900}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
START_RETRIES="${START_RETRIES:-3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_aflow_search}"
RUN_DIR="${RUN_DIR:-${HUB_ROOT}/11_runs/aflow_search/${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(mktemp -d /data/tmp/conifer_aflow_search.XXXXXX)}"

DRY_RUN="${DRY_RUN:-0}"
# Escape hatch for the pre-flight smoke run only: relaxes the 20x20 scale check
# so a 2x4 search can prove the plumbing works.  Never set it for a real run --
# its results are not comparable with the other datasets.
SMOKE_MODE="${SMOKE_MODE:-0}"

fatal() { echo "[fatal] $*" >&2; exit 1; }

validate() {
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" ]] || fatal "python executable missing"
  [[ -s "$DATA_PATH" ]] || fatal "search data missing: $DATA_PATH"
  [[ "$DATA_PATH" != *test* ]] || fatal "refusing to search on a test split: $DATA_PATH"
  [[ -f "$INITIAL_WORKFLOW" ]] || fatal "initial workflow missing: $INITIAL_WORKFLOW"
  [[ "$SMOKE_MODE" == 0 || "$SMOKE_MODE" == 1 ]] || fatal "SMOKE_MODE must be 0 or 1"
  if [[ "$SMOKE_MODE" == 1 ]]; then
    echo "[warn] SMOKE_MODE=1: scale check relaxed (${MAX_ITERATIONS}x${SEARCH_SIZE}); results are NOT reportable"
  else
    [[ "$MAX_ITERATIONS" == 20 && "$SEARCH_SIZE" == 20 ]] || \
      fatal "canonical search requires 20 proposals x 20 tasks (set SMOKE_MODE=1 for a plumbing test)"
  fi
  [[ "$MAX_NEW_TOKENS_EXEC" == 1024 && "$MAX_NEW_TOKENS_OPT" == 2048 && "$MAX_MODEL_LEN" == 16384 ]] || \
    fatal "token budgets must be exec=1024 opt=2048 context=16384"
  [[ "$TEMPERATURE_EXEC" == 0.7 && "$TEMPERATURE_OPT" == 0.8 ]] || \
    fatal "temperatures must be exec=0.7 opt=0.8"
  [[ "$ENABLE_THINKING" == 0 ]] || fatal "Conifer baselines run with thinking disabled"
  local model
  for model in "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" "$MODEL_OPT"; do
    [[ -s "$model/config.json" ]] || fatal "invalid base model: $model"
    [[ ! -e "$model/adapter_config.json" ]] || fatal "LoRA is forbidden: $model"
  done
}

SERVER_PIDS=()
stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}
cleanup() { local code=$?; trap - EXIT INT TERM; stop_servers; exit "$code"; }
trap cleanup EXIT INT TERM

wait_server() {
  local expected="$1" port="$2"
  "$PYTHON_BIN" - "$expected" "http://${HOST}:${port}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json, sys, time, urllib.request
expected, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline, last = time.time() + timeout, None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            names = [str(item["id"]) for item in json.loads(response.read()).get("data", [])]
        if expected in names:
            print(f"server ready: {names}"); raise SystemExit(0)
        last = names
    except SystemExit:
        raise
    except Exception as exc:
        last = exc
    time.sleep(5)
raise SystemExit(f"server timeout: {last}")
PY
}

start_server() {
  local label="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6" attempt runtime
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port" --model "$model" --served-model-name "$served"
    --tensor-parallel-size "$tp" --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --trust-remote-code
  )
  [[ "$ENFORCE_EAGER" == 1 ]] && command+=(--enforce-eager)
  for ((attempt = 1; attempt <= START_RETRIES; attempt++)); do
    runtime="${RUNTIME_ROOT}/${label}_a${attempt}"
    mkdir -p "$runtime/ray"
    echo "[server] $label model=$model gpus=$gpus port=$port attempt=$attempt/$START_RETRIES"
    setsid env CUDA_VISIBLE_DEVICES="$gpus" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      TMPDIR="$runtime" RAY_TMPDIR="$runtime/ray" \
      "${command[@]}" >"${RUN_DIR}/${label}_server.log" 2>&1 &
    local pid="$!"
    SERVER_PIDS+=("$pid")
    if wait_server "$served" "$port"; then return 0; fi
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 5
  done
  fatal "failed to start $label"
}

validate
mkdir -p "$RUN_DIR"

SEARCH_CMD=(
  env "PYTHONPATH=${PYTHONPATH_ROOT}" "$PYTHON_BIN" "${SCRIPT_DIR}/run_search_conifer.py"
  --data-path "$DATA_PATH" --run-dir "$RUN_DIR"
  --initial-workflow "$INITIAL_WORKFLOW"
  --search-size "$SEARCH_SIZE" --search-seed "$SEARCH_SEED" --rng-seed "$RNG_SEED"
  --max-iterations "$MAX_ITERATIONS" --max-concurrency "$MAX_CONCURRENCY"
  --max-new-tokens-executor "$MAX_NEW_TOKENS_EXEC"
  --max-new-tokens-optimizer "$MAX_NEW_TOKENS_OPT"
  --temperature-executor "$TEMPERATURE_EXEC" --temperature-optimizer "$TEMPERATURE_OPT"
  --top-p "$TOP_P" --max-model-len "$MAX_MODEL_LEN" --api-timeout "$API_TIMEOUT"
  --api-base-a1 "http://${HOST}:${A1_PORT}/v1" --api-model-a1 "$SERVED_A1"
  --api-base-a2 "http://${HOST}:${A2_PORT}/v1" --api-model-a2 "$SERVED_A2"
  --api-base-a3 "http://${HOST}:${A3_PORT}/v1" --api-model-a3 "$SERVED_A3"
  --api-base-optimizer "http://${HOST}:${OPT_PORT}/v1" --api-model-optimizer "$SERVED_OPT"
)

if [[ "$DRY_RUN" == 1 ]]; then
  echo "Conifer AFlow search dry-run"
  echo "  run_dir=$RUN_DIR data=$DATA_PATH"
  echo "  search=${MAX_ITERATIONS}x${SEARCH_SIZE} seed=$SEARCH_SEED"
  echo "  exec=${MAX_NEW_TOKENS_EXEC}@${TEMPERATURE_EXEC} opt=${MAX_NEW_TOKENS_OPT}@${TEMPERATURE_OPT} ctx=${MAX_MODEL_LEN}"
  echo "  gpus A1=$A1_GPUS A2=$A2_GPUS A3=$A3_GPUS OPT=$OPT_GPUS"
  echo "  ports $A1_PORT/$A2_PORT/$A3_PORT/$OPT_PORT"
  "${SEARCH_CMD[@]}" --dry-run
  exit 0
fi

exec > >(tee -a "$LOG_PATH") 2>&1

cat >"${RUN_DIR}/config.env" <<EOF
RUN_ID=$RUN_ID
TRAINING_FREE=1
AFLOW_MODE=search
DATA_PATH=$DATA_PATH
INITIAL_WORKFLOW=$INITIAL_WORKFLOW
MAX_ITERATIONS=$MAX_ITERATIONS
SEARCH_SIZE=$SEARCH_SIZE
SEARCH_SEED=$SEARCH_SEED
RNG_SEED=$RNG_SEED
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
MODEL_OPT=$MODEL_OPT
MAX_NEW_TOKENS_EXEC=$MAX_NEW_TOKENS_EXEC
MAX_NEW_TOKENS_OPT=$MAX_NEW_TOKENS_OPT
MAX_MODEL_LEN=$MAX_MODEL_LEN
TEMPERATURE_EXEC=$TEMPERATURE_EXEC
TEMPERATURE_OPT=$TEMPERATURE_OPT
TOP_P=$TOP_P
ENABLE_THINKING=$ENABLE_THINKING
SEARCH_GPU_LAYOUT=2+2+2+2
JSON_TRANSPORT=prompt_only
METRIC=conifer_hard_score_v1
SMOKE_MODE=$SMOKE_MODE
CREATED_AT=$(date '+%F %T %Z')
EOF

start_server A1 "$MODEL_A1" "$SERVED_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP"
start_server A2 "$MODEL_A2" "$SERVED_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP"
start_server A3 "$MODEL_A3" "$SERVED_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP"
start_server OPT "$MODEL_OPT" "$SERVED_OPT" "$OPT_PORT" "$OPT_GPUS" "$OPT_TP"

"${SEARCH_CMD[@]}"

stop_servers
echo "Conifer AFlow search complete: $RUN_DIR"
