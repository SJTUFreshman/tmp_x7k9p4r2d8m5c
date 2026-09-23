#!/usr/bin/env bash
# Run the fixed eight-language zero-shot MAS benchmark as a generation/evaluation pipeline.

set -euo pipefail

SCRIPT_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S %Z')"
SECONDS=0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MULTIPL_E_ROOT="$PROJECT_ROOT/Code/MultiPL-E"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$PYTHON_BIN}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/Code/multipl_e_8lang_benchmark}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/splits}"
SPLIT_SEED="${SPLIT_SEED:-7658190907657085414}"
DATASETS="${DATASETS:-humaneval,mbpp}"
LANGUAGES="${LANGUAGES:-py,cpp,java,php,ts,cs,sh,js}"
HUMANEVAL_MANIFEST="${HUMANEVAL_MANIFEST:-$SPLIT_ROOT/humaneval_train70_test30_seed${SPLIT_SEED}/split_manifest.json}"
MBPP_MANIFEST="${MBPP_MANIFEST:-$SPLIT_ROOT/mbpp_train70_test30_seed${SPLIT_SEED}/split_manifest.json}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8301}"
A2_PORT="${A2_PORT:-8302}"
A3_PORT="${A3_PORT:-8303}"
A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5,6,7}"
A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"

# The MAS generator schedules up to 64 independent trajectories. Completed
# problems are consumed in batches of 64 by 64 isolated evaluator containers.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
PIPELINE_EVAL_BATCH_SIZE="${PIPELINE_EVAL_BATCH_SIZE:-64}"
EVAL_SHARDS="${EVAL_SHARDS:-64}"
EVAL_INNER_WORKERS="${EVAL_INNER_WORKERS:-1}"
PIPELINE_POLL_SECONDS="${PIPELINE_POLL_SECONDS:-0.5}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_object}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-1200}"
ACTION_UNION_NORMALIZATION="handoff_ignores_confirmed_completion_v1"

DOCKER_EXEC="${DOCKER_EXEC:-docker}"
EVAL_IMAGE="${EVAL_IMAGE:-multipl-e-evaluation:jca-current}"
EVAL_QUIET="${EVAL_QUIET:-1}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVALUATOR_PREFLIGHT="${SKIP_EVALUATOR_PREFLIGHT:-0}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ID="${RUN_ID:-${RUN_TIMESTAMP}_multipl_e_8lang_mas_zero_shot_temp${TEMPERATURE//./p}}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/multipl_e_8lang_mas_zero_shot}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/$RUN_ID}"
COMPLETIONS_ROOT="${COMPLETIONS_ROOT:-$MULTIPL_E_ROOT/experiments/multipl_e_8lang_mas_zero_shot}"
COMPLETIONS_DIR="${COMPLETIONS_DIR:-$COMPLETIONS_ROOT/$RUN_ID/completions}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/run.log}"
CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/config.env}"
COMMAND_PATH="${COMMAND_PATH:-$RUN_DIR/command.txt}"
STATUS_PATH="${STATUS_PATH:-$RUN_DIR/run_status.env}"
SCORES_PATH="${SCORES_PATH:-$RUN_DIR/mas_scores.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$RUN_DIR/summary.txt}"
TRAJECTORY_OUTPUT="${TRAJECTORY_OUTPUT:-$RUN_DIR/trajectories.jsonl}"
GENERATION_TIMINGS="${GENERATION_TIMINGS:-$RUN_DIR/generation_timings.jsonl}"
EVALUATION_TIMINGS="${EVALUATION_TIMINGS:-$RUN_DIR/evaluation_timings.jsonl}"
PIPELINE_METRICS="${PIPELINE_METRICS:-$RUN_DIR/pipeline_metrics.env}"
PREFLIGHT_PATH="${PREFLIGHT_PATH:-$RUN_DIR/evaluator_preflight.txt}"
MANIFEST_VALIDATION_PATH="${MANIFEST_VALIDATION_PATH:-$RUN_DIR/manifest_validation.json}"

for path_name in \
    RUN_DIR COMPLETIONS_DIR LOG_PATH CONFIG_PATH COMMAND_PATH STATUS_PATH \
    SCORES_PATH SUMMARY_PATH TRAJECTORY_OUTPUT GENERATION_TIMINGS \
    EVALUATION_TIMINGS PIPELINE_METRICS PREFLIGHT_PATH MANIFEST_VALIDATION_PATH; do
    path_value="${!path_name}"
    if [[ "$path_value" != /* ]]; then
        printf -v "$path_name" '%s/%s' "$PROJECT_ROOT" "$path_value"
    fi
done

require_positive_uint() {
    local name="$1" value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

require_boolean() {
    local name="$1" value="$2"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "ERROR: $name must be 0 or 1, got: $value" >&2
        exit 2
    fi
}

require_file() {
    local label="$1" path="$2"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $label does not exist: $path" >&2
        exit 2
    fi
}

require_model() {
    local agent="$1" path="$2"
    if [[ ! -d "$path" ]]; then
        echo "ERROR: $agent model directory does not exist: $path" >&2
        exit 2
    fi
    if [[ -f "$path/adapter_config.json" && ! -f "$path/config.json" ]]; then
        echo "ERROR: $agent model is a LoRA adapter-only checkpoint: $path" >&2
        exit 2
    fi
    require_file "$agent model config" "$path/config.json"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

for pair in \
    "A1_TP:$A1_TP" \
    "A2_TP:$A2_TP" \
    "A3_TP:$A3_TP" \
    "MAX_MODEL_LEN:$MAX_MODEL_LEN" \
    "SERVER_WAIT_TIMEOUT:$SERVER_WAIT_TIMEOUT" \
    "MAX_CONCURRENCY:$MAX_CONCURRENCY" \
    "PIPELINE_EVAL_BATCH_SIZE:$PIPELINE_EVAL_BATCH_SIZE" \
    "EVAL_SHARDS:$EVAL_SHARDS" \
    "EVAL_INNER_WORKERS:$EVAL_INNER_WORKERS" \
    "T_MAX:$T_MAX" \
    "MAX_NEW_TOKENS:$MAX_NEW_TOKENS"; do
    require_positive_uint "${pair%%:*}" "${pair#*:}"
done
require_boolean EVAL_QUIET "$EVAL_QUIET"
require_boolean KEEP_SERVERS "$KEEP_SERVERS"
require_boolean DRY_RUN "$DRY_RUN"
require_boolean SKIP_EVALUATOR_PREFLIGHT "$SKIP_EVALUATOR_PREFLIGHT"

case "$START_AGENT" in
    A1|A2|A3) ;;
    *) echo "ERROR: START_AGENT must be A1, A2, or A3" >&2; exit 2 ;;
esac
case "$JSON_TRANSPORT" in
    json_object|none) ;;
    *) echo "ERROR: JSON_TRANSPORT must be json_object or none" >&2; exit 2 ;;
esac
if [[ ! "$TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: TEMPERATURE must be a non-negative decimal, got: $TEMPERATURE" >&2
    exit 2
fi
EXPECTED_LANGUAGES="py,cpp,java,php,ts,cs,sh,js"
if [[ "${LANGUAGES// /}" != "$EXPECTED_LANGUAGES" ]]; then
    echo "ERROR: LANGUAGES must be exactly $EXPECTED_LANGUAGES" >&2
    exit 2
fi

MAS_GENERATOR="$PROJECT_ROOT/scripts/run_multipl_e_mas_zero_shot.py"
PIPELINE_RUNNER="$MULTIPL_E_ROOT/scripts/run_multipl_e_pipeline.py"
PARALLEL_EVALUATOR="$MULTIPL_E_ROOT/scripts/evaluate_multipl_e_parallel.py"
SUMMARIZER="$MULTIPL_E_ROOT/scripts/summarize_multipl_e_sas.py"
EVALUATOR_PREFLIGHT="$MULTIPL_E_ROOT/scripts/check_multipl_e_evaluator.py"
MANIFEST_VALIDATOR="$MULTIPL_E_ROOT/scripts/validate_multipl_e_8lang_manifests.py"
require_file "MAS generator" "$MAS_GENERATOR"
require_file "pipeline scheduler" "$PIPELINE_RUNNER"
require_file "parallel evaluator" "$PARALLEL_EVALUATOR"
require_file "result summarizer" "$SUMMARIZER"
require_file "evaluator preflight" "$EVALUATOR_PREFLIGHT"
require_file "manifest validator" "$MANIFEST_VALIDATOR"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
MANIFESTS=()
for dataset in "${DATASET_LIST[@]}"; do
    dataset="${dataset// /}"
    case "$dataset" in
        humaneval) manifest="$HUMANEVAL_MANIFEST" ;;
        mbpp) manifest="$MBPP_MANIFEST" ;;
        *) echo "ERROR: unsupported dataset '$dataset'" >&2; exit 2 ;;
    esac
    require_file "$dataset split manifest" "$manifest"
    MANIFESTS+=("$manifest")
done
if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
    echo "ERROR: DATASETS selected no manifests" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$COMPLETIONS_DIR"
cd "$PROJECT_ROOT"

VALIDATE_CMD=("$PYTHON_BIN" "$MANIFEST_VALIDATOR" --seed "$SPLIT_SEED")
for manifest in "${MANIFESTS[@]}"; do
    VALIDATE_CMD+=(--manifest "$manifest")
done
"${VALIDATE_CMD[@]}" > "$MANIFEST_VALIDATION_PATH"

IFS=',' read -r -a LANGUAGE_LIST <<< "$LANGUAGES"
LANGUAGE_ARGS=()
for language in "${LANGUAGE_LIST[@]}"; do
    LANGUAGE_ARGS+=(--language "${language// /}")
done

GEN_CMD=(
    "$PYTHON_BIN" "$MAS_GENERATOR"
    --split-manifest "${MANIFESTS[@]}"
    --output-dir "$COMPLETIONS_DIR"
    --trajectory-output "$TRAJECTORY_OUTPUT"
    --timings-file "$GENERATION_TIMINGS"
    --t-max "$T_MAX"
    --start-agent "$START_AGENT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-concurrency "$MAX_CONCURRENCY"
    --json-transport "$JSON_TRANSPORT"
    --log-raw-chars "$LOG_RAW_CHARS"
    --api-base-a1 "http://$HOST:$A1_PORT/v1"
    --api-base-a2 "http://$HOST:$A2_PORT/v1"
    --api-base-a3 "http://$HOST:$A3_PORT/v1"
    --api-model-a1 A1
    --api-model-a2 A2
    --api-model-a3 A3
    "${LANGUAGE_ARGS[@]}"
)

PIPELINE_CMD=(
    "$PYTHON_BIN" "$PIPELINE_RUNNER"
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
    --evaluator-script "$PARALLEL_EVALUATOR"
)
if [[ "$EVAL_QUIET" == "1" ]]; then
    PIPELINE_CMD+=(--quiet-evaluator)
fi
PIPELINE_CMD+=(-- "${GEN_CMD[@]}")

SUMMARY_CMD=(
    "$PYTHON_BIN" "$SUMMARIZER"
    --split-manifest "${MANIFESTS[@]}"
    --completions-dir "$COMPLETIONS_DIR"
    --output "$SCORES_PATH"
    --text-output "$SUMMARY_PATH"
    --model-name "$(basename -- "$MODEL_A1")+$(basename -- "$MODEL_A2")+$(basename -- "$MODEL_A3")"
    --evaluation multi-agent-zero-shot
)
for language in "${LANGUAGE_LIST[@]}"; do
    SUMMARY_CMD+=(--language "${language// /}")
done

{
    printf 'RUN_NAME=multipl_e_8lang_mas_zero_shot\n'
    printf 'RUN_ID=%q\nSTARTED_AT=%q\n' "$RUN_ID" "$SCRIPT_STARTED_AT"
    printf 'PROJECT_ROOT=%q\nDATASET_ROOT=%q\n' "$PROJECT_ROOT" "$DATASET_ROOT"
    printf 'MODEL_A1=%q\nMODEL_A2=%q\nMODEL_A3=%q\n' "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"
    printf 'A1_GPUS=%q\nA2_GPUS=%q\nA3_GPUS=%q\n' "$A1_GPUS" "$A2_GPUS" "$A3_GPUS"
    printf 'A1_TP=%q\nA2_TP=%q\nA3_TP=%q\n' "$A1_TP" "$A2_TP" "$A3_TP"
    printf 'HOST=%q\nA1_PORT=%q\nA2_PORT=%q\nA3_PORT=%q\n' "$HOST" "$A1_PORT" "$A2_PORT" "$A3_PORT"
    printf 'DATASETS=%q\nLANGUAGES=%q\nSPLIT_SEED=%q\n' "$DATASETS" "$LANGUAGES" "$SPLIT_SEED"
    printf 'T_MAX=%q\nSTART_AGENT=%q\nMAX_NEW_TOKENS=%q\n' "$T_MAX" "$START_AGENT" "$MAX_NEW_TOKENS"
    printf 'TEMPERATURE=%q\nTOP_P=%q\nMAX_CONCURRENCY=%q\n' "$TEMPERATURE" "$TOP_P" "$MAX_CONCURRENCY"
    printf 'JSON_TRANSPORT=%q\nACTION_UNION_NORMALIZATION=%q\n' "$JSON_TRANSPORT" "$ACTION_UNION_NORMALIZATION"
    printf 'TORCH_DTYPE=%q\nMAX_MODEL_LEN=%q\nGPU_MEMORY_UTILIZATION=%q\n' "$TORCH_DTYPE" "$MAX_MODEL_LEN" "$GPU_MEMORY_UTILIZATION"
    printf 'SERVER_WAIT_TIMEOUT=%q\nPIPELINE_POLL_SECONDS=%q\n' "$SERVER_WAIT_TIMEOUT" "$PIPELINE_POLL_SECONDS"
    printf 'PIPELINE_EVAL_BATCH_SIZE=%q\nEVAL_SHARDS=%q\nEVAL_INNER_WORKERS=%q\n' "$PIPELINE_EVAL_BATCH_SIZE" "$EVAL_SHARDS" "$EVAL_INNER_WORKERS"
    printf 'EVAL_IMAGE=%q\nCOMPLETIONS_DIR=%q\nRUN_DIR=%q\n' "$EVAL_IMAGE" "$COMPLETIONS_DIR" "$RUN_DIR"
    printf 'TRAJECTORY_OUTPUT=%q\nGENERATION_TIMINGS=%q\nEVALUATION_TIMINGS=%q\n' "$TRAJECTORY_OUTPUT" "$GENERATION_TIMINGS" "$EVALUATION_TIMINGS"
    printf 'MANIFEST_VALIDATION_PATH=%q\n' "$MANIFEST_VALIDATION_PATH"
    for manifest in "${MANIFESTS[@]}"; do
        printf 'SPLIT_MANIFEST=%q\n' "$manifest"
    done
} > "$CONFIG_PATH"

{
    printf 'Manifest validation command:\n'
    print_command "${VALIDATE_CMD[@]}"
    printf 'MAS generation command:\n'
    print_command "${GEN_CMD[@]}"
    printf 'Pipeline command:\n'
    print_command "${PIPELINE_CMD[@]}"
    printf 'Summary command:\n'
    print_command "${SUMMARY_CMD[@]}"
} > "$COMMAND_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run passed; manifests validated and commands written to $COMMAND_PATH"
    cat "$COMMAND_PATH"
    exit 0
fi

require_model A1 "$MODEL_A1"
require_model A2 "$MODEL_A2"
require_model A3 "$MODEL_A3"

SERVER_PIDS=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$KEEP_SERVERS" != "1" ]]; then
        for pid in "${SERVER_PIDS[@]}"; do
            kill -TERM "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
        done
        if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
            sleep 5
        fi
        for pid in "${SERVER_PIDS[@]}"; do
            kill -KILL "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
        done
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
    "$PYTHON_BIN" - "$1" "$2" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json
import sys
import time
import urllib.request

name, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
        model_ids = [item.get("id") for item in data.get("data", [])]
        if name in model_ids:
            print(f"{name} ready: {model_ids}")
            raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit(f"{name} did not become ready within {timeout:g} seconds")
PY
}

start_server() {
    local agent="$1" model="$2" port="$3" gpus="$4" tp="$5" server_log="$6"
    setsid env CUDA_VISIBLE_DEVICES="$gpus" "$VLLM_PYTHON_BIN" \
        -m vllm.entrypoints.openai.api_server \
        --host "$HOST" \
        --port "$port" \
        --model "$model" \
        --served-model-name "$agent" \
        --tensor-parallel-size "$tp" \
        --dtype "$TORCH_DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code > "$server_log" 2>&1 &
    SERVER_PIDS+=("$!")
    echo "Starting $agent on GPUs $gpus, TP=$tp, port=$port; log=$server_log"
}

if [[ "$SKIP_EVALUATOR_PREFLIGHT" != "1" ]]; then
    set +e
    "$PYTHON_BIN" "$EVALUATOR_PREFLIGHT" \
        --image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC" \
        2>&1 | tee "$PREFLIGHT_PATH"
    PREFLIGHT_STATUS=${PIPESTATUS[0]}
    set -e
    if [[ "$PREFLIGHT_STATUS" -ne 0 ]]; then
        exit "$PREFLIGHT_STATUS"
    fi
fi

exec > >(tee "$LOG_PATH") 2>&1
echo "===== MultiPL-E eight-language MAS zero-shot pipeline ====="
echo "Started: $SCRIPT_STARTED_AT"
echo "Run ID: $RUN_ID"
echo "Models: A1=$MODEL_A1 A2=$MODEL_A2 A3=$MODEL_A3"
echo "Datasets: $DATASETS"
echo "Languages: $LANGUAGES"
echo "Split seed: $SPLIT_SEED"
echo "T max: $T_MAX"
echo "MAS concurrency: $MAX_CONCURRENCY"
echo "Evaluation batch/shards: $PIPELINE_EVAL_BATCH_SIZE/$EVAL_SHARDS"

start_server A1 "$MODEL_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$RUN_DIR/A1_server.log"
start_server A2 "$MODEL_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$RUN_DIR/A2_server.log"
start_server A3 "$MODEL_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$RUN_DIR/A3_server.log"
wait_for_server A1 "http://$HOST:$A1_PORT/v1/models"
wait_for_server A2 "http://$HOST:$A2_PORT/v1/models"
wait_for_server A3 "http://$HOST:$A3_PORT/v1/models"

STATUS=0
set +e
"${PIPELINE_CMD[@]}"
STATUS=$?
set -e
if [[ "$STATUS" -eq 0 ]]; then
    set +e
    "${SUMMARY_CMD[@]}"
    STATUS=$?
    set -e
fi

GENERATION_SECONDS=0
EVALUATION_WORK_SECONDS=0
if [[ -f "$PIPELINE_METRICS" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            GENERATION_SECONDS) GENERATION_SECONDS="$value" ;;
            EVALUATION_WORK_SECONDS) EVALUATION_WORK_SECONDS="$value" ;;
        esac
    done < "$PIPELINE_METRICS"
fi
COMPLETED_AT="$(date '+%Y-%m-%d %H:%M:%S %Z')"
{
    printf 'STATUS=%s\n' "$STATUS"
    printf 'GENERATION_SECONDS=%s\nEVALUATION_WORK_SECONDS=%s\n' "$GENERATION_SECONDS" "$EVALUATION_WORK_SECONDS"
    printf 'ELAPSED_SECONDS=%s\n' "$SECONDS"
    printf 'RUN_ID=%q\nSTARTED_AT=%q\nCOMPLETED_AT=%q\n' "$RUN_ID" "$SCRIPT_STARTED_AT" "$COMPLETED_AT"
    printf 'RUN_DIR=%q\nCOMPLETIONS_DIR=%q\nSCORES_PATH=%q\n' "$RUN_DIR" "$COMPLETIONS_DIR" "$SCORES_PATH"
} > "$STATUS_PATH"

echo "===== Run completion ====="
echo "Status: $STATUS"
echo "Generation seconds: $GENERATION_SECONDS"
echo "Evaluation work seconds: $EVALUATION_WORK_SECONDS"
echo "Total elapsed seconds: $SECONDS"
echo "Scores: $SCORES_PATH"
echo "Summary: $SUMMARY_PATH"
exit "$STATUS"
