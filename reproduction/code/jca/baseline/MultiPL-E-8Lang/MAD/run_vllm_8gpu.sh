#!/usr/bin/env bash
# Training-free MAD on MultiPL-E-8Lang with three pure Qwen3 base models.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
MULTIPL_E_ROOT="$PROJECT_ROOT/Code/MultiPL-E"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$PYTHON_BIN}"
VLLM_COMPAT_DIR="${VLLM_COMPAT_DIR:-$PROJECT_ROOT/baseline/MuSiQue/vllm_compat}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/Code/multipl_e_8lang_benchmark}"
SPLIT_SEED="${SPLIT_SEED:-7658190907657085414}"
DATASETS="${DATASETS:-humaneval,mbpp}"
LANGUAGES="${LANGUAGES:-py,cpp,java,php,ts,cs,sh,js}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/splits}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"
A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"
A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5,6,7}"
A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-4}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
N_ROUNDS="${N_ROUNDS:-3}"
TEMPERATURE_ROUND0="${TEMPERATURE_ROUND0:-0.9}"
TEMPERATURE_DEBATE="${TEMPERATURE_DEBATE:-0.3}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-A3,A2,A1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
MAX_PROBLEMS_PER_LANGUAGE="${MAX_PROBLEMS_PER_LANGUAGE:-}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"

EVAL_IMAGE="${EVAL_IMAGE:-multipl-e-evaluation:jca-current}"
DOCKER_EXEC="${DOCKER_EXEC:-docker}"
EVAL_SHARDS="${EVAL_SHARDS:-64}"
EVAL_INNER_WORKERS="${EVAL_INNER_WORKERS:-1}"
PIPELINE_EVAL_BATCH_SIZE="${PIPELINE_EVAL_BATCH_SIZE:-64}"
PIPELINE_POLL_SECONDS="${PIPELINE_POLL_SECONDS:-0.5}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVALUATOR_PREFLIGHT="${SKIP_EVALUATOR_PREFLIGHT:-0}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_multipl_e_8lang_mad_r${N_ROUNDS}}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/multipl_e_8lang_baselines/mad}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/$RUN_ID}"
COMPLETIONS_DIR="${COMPLETIONS_DIR:-$MULTIPL_E_ROOT/experiments/multipl_e_8lang_baselines/mad/$RUN_ID/completions}"
TRAJECTORY_OUTPUT="${TRAJECTORY_OUTPUT:-$RUN_DIR/trajectories.jsonl}"
GENERATION_TIMINGS="${GENERATION_TIMINGS:-$RUN_DIR/generation_timings.jsonl}"
EVALUATION_TIMINGS="${EVALUATION_TIMINGS:-$RUN_DIR/evaluation_timings.jsonl}"
PIPELINE_METRICS="${PIPELINE_METRICS:-$RUN_DIR/pipeline_metrics.env}"
SCORES_PATH="${SCORES_PATH:-$RUN_DIR/mas_scores.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$RUN_DIR/summary.txt}"
MAD_METRICS_PATH="${MAD_METRICS_PATH:-$RUN_DIR/mad_metrics.json}"
MAD_SUMMARY_PATH="${MAD_SUMMARY_PATH:-$RUN_DIR/mad_summary.txt}"

RUNNER="$SCRIPT_DIR/run_mad.py"
MAD_SUMMARIZER="$SCRIPT_DIR/summarize_mad.py"
PIPELINE="$MULTIPL_E_ROOT/scripts/run_multipl_e_pipeline.py"
EVALUATOR="$MULTIPL_E_ROOT/scripts/evaluate_multipl_e_parallel.py"
SUMMARIZER="$MULTIPL_E_ROOT/scripts/summarize_multipl_e_sas.py"
PREFLIGHT="$MULTIPL_E_ROOT/scripts/check_multipl_e_evaluator.py"

mkdir -p "$RUN_DIR" "$COMPLETIONS_DIR"
exec > >(tee "$RUN_DIR/run.log") 2>&1

fatal() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$2" ]] || fatal "$1 missing: $2"; }
require_dir() { [[ -d "$2" ]] || fatal "$1 missing: $2"; }
require_bool() { [[ "$2" == 0 || "$2" == 1 ]] || fatal "$1 must be 0 or 1"; }

require_bool ENABLE_THINKING "$ENABLE_THINKING"
require_bool USE_EXISTING_SERVERS "$USE_EXISTING_SERVERS"
require_bool KEEP_SERVERS "$KEEP_SERVERS"
require_bool DRY_RUN "$DRY_RUN"
require_bool SKIP_EVALUATOR_PREFLIGHT "$SKIP_EVALUATOR_PREFLIGHT"
[[ "$ENABLE_THINKING" == 0 ]] || fatal "MultiPL-E-8Lang MAD requires ENABLE_THINKING=0"
[[ "$TIE_BREAK_ORDER" == A3,A2,A1 ]] || fatal "TIE_BREAK_ORDER must be A3,A2,A1"
[[ "$LANGUAGES" == py,cpp,java,php,ts,cs,sh,js ]] || fatal "LANGUAGES must be py,cpp,java,php,ts,cs,sh,js"
[[ "$N_ROUNDS" =~ ^[1-9][0-9]*$ ]] || fatal "N_ROUNDS must be positive"
for pair in runner:$RUNNER mad-summarizer:$MAD_SUMMARIZER pipeline:$PIPELINE evaluator:$EVALUATOR summarizer:$SUMMARIZER evaluator-preflight:$PREFLIGHT; do
  require_file "${pair%%:*}" "${pair#*:}"
done

HUMANEVAL_MANIFEST="$SPLIT_ROOT/humaneval_train70_test30_seed${SPLIT_SEED}/split_manifest.json"
MBPP_MANIFEST="$SPLIT_ROOT/mbpp_train70_test30_seed${SPLIT_SEED}/split_manifest.json"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
MANIFESTS=()
for dataset in "${DATASET_LIST[@]}"; do
  case "${dataset// /}" in
    humaneval) MANIFESTS+=("$HUMANEVAL_MANIFEST") ;;
    mbpp) MANIFESTS+=("$MBPP_MANIFEST") ;;
    *) fatal "unsupported dataset: $dataset" ;;
  esac
done
for manifest in "${MANIFESTS[@]}"; do require_file manifest "$manifest"; done

SERVER_PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_SERVERS" != 1 ]]; then
    for pid in "${SERVER_PIDS[@]}"; do
      kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM -- "$pid" >/dev/null 2>&1 || true
    done
    sleep 5
    for pid in "${SERVER_PIDS[@]}"; do
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL -- "$pid" >/dev/null 2>&1 || true
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_model() {
  "$PYTHON_BIN" - "$1" "$2" "$3" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json, sys, time, urllib.request
name, url, expected, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = [item.get("id") for item in json.load(response).get("data", [])]
        if expected in models:
            print(f"{name} ready: {models}")
            raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit(f"{name} did not expose {expected!r} before timeout")
PY
}

start_server() {
  local agent="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6"
  require_dir "$agent model" "$model"
  [[ -f "$model/config.json" ]] || fatal "$agent base model config missing: $model/config.json"
  [[ ! -f "$model/adapter_config.json" ]] || fatal "$agent model path is a LoRA adapter: $model"
  setsid env CUDA_VISIBLE_DEVICES="$gpus" \
    PYTHONPATH="${VLLM_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --host "$HOST" --port "$port" --model "$model" \
      --served-model-name "$served" --tensor-parallel-size "$tp" \
      --dtype "$TORCH_DTYPE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
      >"$RUN_DIR/${agent}_server.log" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "Starting $agent model=$model served=$served GPUs=$gpus TP=$tp port=$port"
}

IFS=',' read -r -a LANGUAGE_LIST <<< "$LANGUAGES"
LANGUAGE_ARGS=()
for language in "${LANGUAGE_LIST[@]}"; do LANGUAGE_ARGS+=(--language "$language"); done

GEN_CMD=(
  "$PYTHON_BIN" "$RUNNER"
  --split-manifest "${MANIFESTS[@]}"
  --output-dir "$COMPLETIONS_DIR"
  --trajectory-output "$TRAJECTORY_OUTPUT"
  --timings-file "$GENERATION_TIMINGS"
  --n-rounds "$N_ROUNDS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature-round0 "$TEMPERATURE_ROUND0"
  --temperature-debate "$TEMPERATURE_DEBATE"
  --top-p "$TOP_P"
  --api-timeout "$API_TIMEOUT"
  --max-concurrency "$MAX_CONCURRENCY"
  --log-raw-chars "$LOG_RAW_CHARS"
  --api-base-a1 "http://$HOST:$A1_PORT/v1"
  --api-base-a2 "http://$HOST:$A2_PORT/v1"
  --api-base-a3 "http://$HOST:$A3_PORT/v1"
  --api-model-a1 "$SERVED_A1"
  --api-model-a2 "$SERVED_A2"
  --api-model-a3 "$SERVED_A3"
  "${LANGUAGE_ARGS[@]}"
)
if [[ -n "$MAX_PROBLEMS_PER_LANGUAGE" ]]; then
  GEN_CMD+=(--max-problems-per-language "$MAX_PROBLEMS_PER_LANGUAGE")
fi

PIPELINE_CMD=(
  "$PYTHON_BIN" "$PIPELINE"
  --input-dir "$COMPLETIONS_DIR"
  --expected-completions 1
  --eval-image "$EVAL_IMAGE"
  --docker-exec "$DOCKER_EXEC"
  --eval-shards "$EVAL_SHARDS"
  --eval-inner-workers "$EVAL_INNER_WORKERS"
  --eval-batch-size "$PIPELINE_EVAL_BATCH_SIZE"
  --poll-interval "$PIPELINE_POLL_SECONDS"
  --metrics-file "$PIPELINE_METRICS"
  --timings-file "$EVALUATION_TIMINGS"
  --evaluator-script "$EVALUATOR"
  --quiet-evaluator
  -- "${GEN_CMD[@]}"
)

{
  printf 'RUN_ID=%q\nRUN_DIR=%q\nCOMPLETIONS_DIR=%q\n' "$RUN_ID" "$RUN_DIR" "$COMPLETIONS_DIR"
  printf 'MODEL_A1=%q\nMODEL_A2=%q\nMODEL_A3=%q\n' "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"
  printf 'TRAINING_FREE=1\nENABLE_THINKING=0\nTIE_BREAK_ORDER=A3,A2,A1\n'
  printf 'N_ROUNDS=%q\nTEMPERATURE_ROUND0=%q\nTEMPERATURE_DEBATE=%q\n' "$N_ROUNDS" "$TEMPERATURE_ROUND0" "$TEMPERATURE_DEBATE"
  printf 'DATASETS=%q\nLANGUAGES=%q\nSPLIT_SEED=%q\n' "$DATASETS" "$LANGUAGES" "$SPLIT_SEED"
  printf 'MAX_NEW_TOKENS=%q\nMAX_CONCURRENCY=%q\nMAX_MODEL_LEN=%q\n' "$MAX_NEW_TOKENS" "$MAX_CONCURRENCY" "$MAX_MODEL_LEN"
  printf 'A1_GPUS=%q\nA2_GPUS=%q\nA3_GPUS=%q\nA1_TP=%q\nA2_TP=%q\nA3_TP=%q\n' "$A1_GPUS" "$A2_GPUS" "$A3_GPUS" "$A1_TP" "$A2_TP" "$A3_TP"
  for manifest in "${MANIFESTS[@]}"; do printf 'SPLIT_MANIFEST=%q\n' "$manifest"; done
} > "$RUN_DIR/config.env"
printf '%q ' "${GEN_CMD[@]}" > "$RUN_DIR/command.txt"; printf '\n' >> "$RUN_DIR/command.txt"
printf '%q ' "${PIPELINE_CMD[@]}" >> "$RUN_DIR/command.txt"; printf '\n' >> "$RUN_DIR/command.txt"

echo "================ MAD MultiPL-E-8Lang ================"
echo "run_id=$RUN_ID training_free=1 thinking=disabled n_rounds=$N_ROUNDS"
echo "A1=$MODEL_A1 A2=$MODEL_A2 A3=$MODEL_A3"
echo "aggregation=normalized_exact_majority tie_break=A3,A2,A1"

if [[ "$DRY_RUN" == 1 ]]; then
  "${GEN_CMD[@]}" --dry-run
  echo "DRY_RUN=1; no servers or evaluator started"
  exit 0
fi

if [[ "$SKIP_EVALUATOR_PREFLIGHT" != 1 ]]; then
  "$PYTHON_BIN" "$PREFLIGHT" --image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC" \
    > "$RUN_DIR/evaluator_preflight.txt"
fi
if [[ "$USE_EXISTING_SERVERS" != 1 ]]; then
  start_server A1 "$MODEL_A1" "$SERVED_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP"
  start_server A2 "$MODEL_A2" "$SERVED_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP"
  start_server A3 "$MODEL_A3" "$SERVED_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP"
fi
wait_for_model A1 "http://$HOST:$A1_PORT/v1/models" "$SERVED_A1"
wait_for_model A2 "http://$HOST:$A2_PORT/v1/models" "$SERVED_A2"
wait_for_model A3 "http://$HOST:$A3_PORT/v1/models" "$SERVED_A3"

"${PIPELINE_CMD[@]}"

SUMMARY_CMD=(
  "$PYTHON_BIN" "$SUMMARIZER"
  --split-manifest "${MANIFESTS[@]}"
  --completions-dir "$COMPLETIONS_DIR"
  --output "$SCORES_PATH"
  --text-output "$SUMMARY_PATH"
  --model-name "MAD:Qwen3-1.7B+4B+8B"
  --evaluation "mad-training-free"
  "${LANGUAGE_ARGS[@]}"
)
if [[ -n "$MAX_PROBLEMS_PER_LANGUAGE" ]]; then SUMMARY_CMD+=(--allow-partial); fi
"${SUMMARY_CMD[@]}"
"$PYTHON_BIN" "$MAD_SUMMARIZER" \
  --trajectories "$TRAJECTORY_OUTPUT" \
  --output "$MAD_METRICS_PATH" \
  --text-output "$MAD_SUMMARY_PATH"

echo "===== Run completion ====="
echo "Scores: $SCORES_PATH"
echo "MAD diagnostics: $MAD_METRICS_PATH"
