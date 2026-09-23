#!/usr/bin/env bash
# AFlow workflow search on train70 followed by frozen-workflow test30 eval.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
MULTIPL_E_ROOT="$PROJECT_ROOT/Code/MultiPL-E"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$PYTHON_BIN}"
VLLM_COMPAT_DIR="${VLLM_COMPAT_DIR:-$PROJECT_ROOT/baseline/MuSiQue/vllm_compat}"

MODE="${MODE:-both}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/Code/multipl_e_8lang_benchmark}"
SPLIT_SEED="${SPLIT_SEED:-7658190907657085414}"
DATASETS="${DATASETS:-humaneval,mbpp}"
LANGUAGES="${LANGUAGES:-py,cpp,java,php,ts,cs,sh,js}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/splits}"

MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
SEARCH_SIZE="${SEARCH_SIZE:-20}"
SEARCH_SEED="${SEARCH_SEED:-20260810}"
RNG_SEED="${RNG_SEED:-20260810}"
INITIAL_WORKFLOW="${INITIAL_WORKFLOW:-$SCRIPT_DIR/workflows/round_00_initial.py}"
WORKFLOW_FILE="${WORKFLOW_FILE:-$INITIAL_WORKFLOW}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_OPT="${MODEL_OPT:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_OPT="${SERVED_OPT:-Optimizer_14B}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"; A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"; OPT_PORT="${OPT_PORT:-8204}"
A1_GPUS="${A1_GPUS:-0,1}"; A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5}"; OPT_GPUS="${OPT_GPUS:-6,7}"
A1_TP="${A1_TP:-2}"; A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-2}"; OPT_TP="${OPT_TP:-2}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NEW_TOKENS_EXEC="${MAX_NEW_TOKENS_EXEC:-4096}"
MAX_NEW_TOKENS_OPT="${MAX_NEW_TOKENS_OPT:-2048}"
TEMPERATURE_EXEC="${TEMPERATURE_EXEC:-0.7}"
TEMPERATURE_OPT="${TEMPERATURE_OPT:-0.8}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
MAX_PROBLEMS_PER_LANGUAGE="${MAX_PROBLEMS_PER_LANGUAGE:-}"
API_TIMEOUT="${API_TIMEOUT:-900}"

EVAL_IMAGE="${EVAL_IMAGE:-multipl-e-evaluation:jca-current}"
DOCKER_EXEC="${DOCKER_EXEC:-docker}"
SEARCH_EVAL_SHARDS="${SEARCH_EVAL_SHARDS:-20}"
EVAL_SHARDS="${EVAL_SHARDS:-64}"
EVAL_INNER_WORKERS="${EVAL_INNER_WORKERS:-1}"
PIPELINE_EVAL_BATCH_SIZE="${PIPELINE_EVAL_BATCH_SIZE:-64}"
PIPELINE_POLL_SECONDS="${PIPELINE_POLL_SECONDS:-0.5}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_MODE="${SMOKE_MODE:-0}"
SKIP_EVALUATOR_PREFLIGHT="${SKIP_EVALUATOR_PREFLIGHT:-0}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_multipl_e_8lang_aflow}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/multipl_e_8lang_baselines/aflow}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/$RUN_ID}"
SEARCH_RUN_DIR="${SEARCH_RUN_DIR:-$RUN_DIR/search}"
COMPLETIONS_DIR="${COMPLETIONS_DIR:-$MULTIPL_E_ROOT/experiments/multipl_e_8lang_baselines/aflow/$RUN_ID/completions}"
TRAJECTORY_OUTPUT="${TRAJECTORY_OUTPUT:-$RUN_DIR/trajectories.jsonl}"
GENERATION_TIMINGS="${GENERATION_TIMINGS:-$RUN_DIR/generation_timings.jsonl}"
EVALUATION_TIMINGS="${EVALUATION_TIMINGS:-$RUN_DIR/evaluation_timings.jsonl}"
PIPELINE_METRICS="${PIPELINE_METRICS:-$RUN_DIR/pipeline_metrics.env}"
SCORES_PATH="${SCORES_PATH:-$RUN_DIR/mas_scores.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$RUN_DIR/summary.txt}"

SEARCH_RUNNER="$SCRIPT_DIR/run_search.py"
EVAL_RUNNER="$SCRIPT_DIR/run_eval.py"
PIPELINE="$MULTIPL_E_ROOT/scripts/run_multipl_e_pipeline.py"
EVALUATOR="$MULTIPL_E_ROOT/scripts/evaluate_multipl_e_parallel.py"
SUMMARIZER="$MULTIPL_E_ROOT/scripts/summarize_multipl_e_sas.py"
PREFLIGHT="$MULTIPL_E_ROOT/scripts/check_multipl_e_evaluator.py"

mkdir -p "$RUN_DIR" "$SEARCH_RUN_DIR" "$COMPLETIONS_DIR"
exec > >(tee "$RUN_DIR/run.log") 2>&1

fatal() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$2" ]] || fatal "$1 missing: $2"; }
require_dir() { [[ -d "$2" ]] || fatal "$1 missing: $2"; }
require_bool() { [[ "$2" == 0 || "$2" == 1 ]] || fatal "$1 must be 0 or 1"; }

case "$MODE" in search|eval|both) ;; *) fatal "MODE must be search, eval, or both" ;; esac
for pair in ENABLE_THINKING:$ENABLE_THINKING USE_EXISTING_SERVERS:$USE_EXISTING_SERVERS KEEP_SERVERS:$KEEP_SERVERS DRY_RUN:$DRY_RUN SMOKE_MODE:$SMOKE_MODE SKIP_EVALUATOR_PREFLIGHT:$SKIP_EVALUATOR_PREFLIGHT; do
  require_bool "${pair%%:*}" "${pair#*:}"
done
[[ "$ENABLE_THINKING" == 0 ]] || fatal "MultiPL-E AFlow requires ENABLE_THINKING=0"
if [[ "$SMOKE_MODE" != 1 ]]; then
  [[ "$MAX_ITERATIONS" == 20 ]] || fatal "canonical search requires MAX_ITERATIONS=20"
  [[ "$SEARCH_SIZE" == 20 ]] || fatal "canonical search requires SEARCH_SIZE=20"
fi
[[ "$LANGUAGES" == py,cpp,java,php,ts,cs,sh,js ]] || fatal "LANGUAGES must contain the canonical eight languages"
for pair in search-runner:$SEARCH_RUNNER eval-runner:$EVAL_RUNNER pipeline:$PIPELINE evaluator:$EVALUATOR summarizer:$SUMMARIZER preflight:$PREFLIGHT initial-workflow:$INITIAL_WORKFLOW; do
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
IFS=',' read -r -a LANGUAGE_LIST <<< "$LANGUAGES"
LANGUAGE_ARGS=()
for language in "${LANGUAGE_LIST[@]}"; do LANGUAGE_ARGS+=(--language "$language"); done

SERVER_PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_SERVERS" != 1 ]]; then
    for pid in "${SERVER_PIDS[@]}"; do
      kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM -- "$pid" >/dev/null 2>&1 || true
    done
    if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then sleep 5; fi
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
  local gpu_count
  gpu_count="$(awk -F',' '{print NF}' <<< "$gpus")"
  [[ "$gpu_count" == "$tp" ]] || fatal "$agent GPU count $gpu_count must equal TP $tp"
  setsid env CUDA_VISIBLE_DEVICES="$gpus" \
    PYTHONPATH="${VLLM_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --host "$HOST" --port "$port" --model "$model" \
      --served-model-name "$served" --tensor-parallel-size "$tp" \
      --dtype "$TORCH_DTYPE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
      >"$RUN_DIR/${agent}_server.log" 2>&1 &
  SERVER_PIDS+=("$!")
}

SEARCH_CMD=(
  "$PYTHON_BIN" "$SEARCH_RUNNER"
  --split-manifest "${MANIFESTS[@]}" --run-dir "$SEARCH_RUN_DIR"
  --initial-workflow "$INITIAL_WORKFLOW"
  --search-size "$SEARCH_SIZE" --search-seed "$SEARCH_SEED"
  --max-iterations "$MAX_ITERATIONS" --rng-seed "$RNG_SEED"
  --max-new-tokens-executor "$MAX_NEW_TOKENS_EXEC"
  --max-new-tokens-optimizer "$MAX_NEW_TOKENS_OPT"
  --temperature-executor "$TEMPERATURE_EXEC"
  --temperature-optimizer "$TEMPERATURE_OPT" --top-p "$TOP_P"
  --api-base-a1 "http://$HOST:$A1_PORT/v1" --api-model-a1 "$SERVED_A1"
  --api-base-a2 "http://$HOST:$A2_PORT/v1" --api-model-a2 "$SERVED_A2"
  --api-base-a3 "http://$HOST:$A3_PORT/v1" --api-model-a3 "$SERVED_A3"
  --api-base-optimizer "http://$HOST:$OPT_PORT/v1" --api-model-optimizer "$SERVED_OPT"
  --api-timeout "$API_TIMEOUT" --max-concurrency "$MAX_CONCURRENCY"
  --evaluator-script "$EVALUATOR" --eval-image "$EVAL_IMAGE"
  --docker-exec "$DOCKER_EXEC" --eval-shards "$SEARCH_EVAL_SHARDS"
  --eval-inner-workers "$EVAL_INNER_WORKERS" "${LANGUAGE_ARGS[@]}"
)
if [[ "$SMOKE_MODE" == 1 ]]; then SEARCH_CMD+=(--smoke-mode); fi

find_best_workflow() {
  "$PYTHON_BIN" - "$SEARCH_RUN_DIR/state.json" <<'PY'
import json, sys
nodes = json.load(open(sys.argv[1], encoding="utf-8"))
valid = [node for node in nodes if node.get("parse_ok")]
if not valid:
    raise SystemExit("No valid AFlow workflow")
print(max(valid, key=lambda node: (node["dev_em"], node["dev_f1"]))["source_file"])
PY
}

run_test_eval() {
  local workflow_file="$1"
  require_file workflow "$workflow_file"
  local gen_cmd=(
    "$PYTHON_BIN" "$EVAL_RUNNER"
    --split-manifest "${MANIFESTS[@]}" --workflow-file "$workflow_file"
    --output-dir "$COMPLETIONS_DIR" --trajectory-output "$TRAJECTORY_OUTPUT"
    --timings-file "$GENERATION_TIMINGS" --max-new-tokens "$MAX_NEW_TOKENS_EXEC"
    --temperature "$TEMPERATURE_EXEC" --top-p "$TOP_P"
    --api-base-a1 "http://$HOST:$A1_PORT/v1" --api-model-a1 "$SERVED_A1"
    --api-base-a2 "http://$HOST:$A2_PORT/v1" --api-model-a2 "$SERVED_A2"
    --api-base-a3 "http://$HOST:$A3_PORT/v1" --api-model-a3 "$SERVED_A3"
    --api-timeout "$API_TIMEOUT" --max-concurrency "$MAX_CONCURRENCY"
    "${LANGUAGE_ARGS[@]}"
  )
  if [[ -n "$MAX_PROBLEMS_PER_LANGUAGE" ]]; then
    gen_cmd+=(--max-problems-per-language "$MAX_PROBLEMS_PER_LANGUAGE")
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    "${gen_cmd[@]}" --dry-run
    return
  fi
  local pipeline_cmd=(
    "$PYTHON_BIN" "$PIPELINE" --input-dir "$COMPLETIONS_DIR"
    --expected-completions 1 --eval-image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC"
    --eval-shards "$EVAL_SHARDS" --eval-inner-workers "$EVAL_INNER_WORKERS"
    --eval-batch-size "$PIPELINE_EVAL_BATCH_SIZE" --poll-interval "$PIPELINE_POLL_SECONDS"
    --metrics-file "$PIPELINE_METRICS" --timings-file "$EVALUATION_TIMINGS"
    --evaluator-script "$EVALUATOR" --quiet-evaluator -- "${gen_cmd[@]}"
  )
  printf '%q ' "${pipeline_cmd[@]}" >> "$RUN_DIR/command.txt"; printf '\n' >> "$RUN_DIR/command.txt"
  "${pipeline_cmd[@]}"
  local summary_cmd=(
    "$PYTHON_BIN" "$SUMMARIZER" --split-manifest "${MANIFESTS[@]}"
    --completions-dir "$COMPLETIONS_DIR" --output "$SCORES_PATH"
    --text-output "$SUMMARY_PATH" --model-name "AFlow:Qwen3-1.7B+4B+8B+Opt14B"
    --evaluation "aflow-train70-search-test30-eval" "${LANGUAGE_ARGS[@]}"
  )
  if [[ -n "$MAX_PROBLEMS_PER_LANGUAGE" ]]; then summary_cmd+=(--allow-partial); fi
  "${summary_cmd[@]}"
}

{
  printf 'RUN_ID=%q\nMODE=%q\nRUN_DIR=%q\nSEARCH_RUN_DIR=%q\n' "$RUN_ID" "$MODE" "$RUN_DIR" "$SEARCH_RUN_DIR"
  printf 'SEARCH_PARTITION=train70\nFINAL_PARTITION=test30\nMAX_ITERATIONS=%q\nSEARCH_SIZE=%q\nSMOKE_MODE=%q\n' "$MAX_ITERATIONS" "$SEARCH_SIZE" "$SMOKE_MODE"
  printf 'MODEL_A1=%q\nMODEL_A2=%q\nMODEL_A3=%q\nMODEL_OPT=%q\n' "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" "$MODEL_OPT"
  printf 'TRAINING_FREE=1\nENABLE_THINKING=0\nTEST30_FEEDBACK_TO_SEARCH=0\n'
} > "$RUN_DIR/config.env"
printf '%q ' "${SEARCH_CMD[@]}" > "$RUN_DIR/command.txt"; printf '\n' >> "$RUN_DIR/command.txt"

echo "================ AFlow MultiPL-E-8Lang ================"
echo "mode=$MODE search=train70:${SEARCH_SIZE}tasks:${MAX_ITERATIONS}iterations final=test30"
echo "A1=$MODEL_A1 A2=$MODEL_A2 A3=$MODEL_A3 OPT=$MODEL_OPT thinking=disabled"

if [[ "$DRY_RUN" == 1 ]]; then
  if [[ "$MODE" == search || "$MODE" == both ]]; then "${SEARCH_CMD[@]}" --dry-run; fi
  if [[ "$MODE" == eval || "$MODE" == both ]]; then run_test_eval "$WORKFLOW_FILE"; fi
  echo "DRY_RUN=1; no servers or Docker evaluator started"
  exit 0
fi

if [[ "$SKIP_EVALUATOR_PREFLIGHT" != 1 ]]; then
  "$PYTHON_BIN" "$PREFLIGHT" --image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC" > "$RUN_DIR/evaluator_preflight.txt"
fi
if [[ "$USE_EXISTING_SERVERS" != 1 ]]; then
  start_server A1 "$MODEL_A1" "$SERVED_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP"
  start_server A2 "$MODEL_A2" "$SERVED_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP"
  start_server A3 "$MODEL_A3" "$SERVED_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP"
  if [[ "$MODE" == search || "$MODE" == both ]]; then
    start_server OPT "$MODEL_OPT" "$SERVED_OPT" "$OPT_PORT" "$OPT_GPUS" "$OPT_TP"
  fi
fi
wait_for_model A1 "http://$HOST:$A1_PORT/v1/models" "$SERVED_A1"
wait_for_model A2 "http://$HOST:$A2_PORT/v1/models" "$SERVED_A2"
wait_for_model A3 "http://$HOST:$A3_PORT/v1/models" "$SERVED_A3"
if [[ "$MODE" == search || "$MODE" == both ]]; then
  wait_for_model OPT "http://$HOST:$OPT_PORT/v1/models" "$SERVED_OPT"
  "${SEARCH_CMD[@]}"
  WORKFLOW_FILE="$(find_best_workflow)"
  echo "Frozen best workflow: $WORKFLOW_FILE"
fi
if [[ "$MODE" == eval || "$MODE" == both ]]; then
  run_test_eval "$WORKFLOW_FILE"
fi

echo "AFlow run complete: $RUN_DIR"
