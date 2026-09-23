#!/usr/bin/env bash
# Run the fixed eight-language SAS benchmark with overlapped generation/evaluation.

set -euo pipefail

SCRIPT_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S %Z')"
SECONDS=0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/Code/MultiPL-E"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/data/wangyuheng/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
NUM_GPUS="${NUM_GPUS:-8}"

DATASET_ROOT="${DATASET_ROOT:-/data/wangyuheng/jca/Code/multipl_e_8lang_benchmark}"
SPLIT_ROOT="${SPLIT_ROOT:-$DATASET_ROOT/splits}"
SPLIT_SEED="${SPLIT_SEED:-7658190907657085414}"
DATASETS="${DATASETS:-humaneval,mbpp}"
LANGUAGES="${LANGUAGES:-py,cpp,java,php,ts,cs,sh,js}"
HUMANEVAL_MANIFEST="${HUMANEVAL_MANIFEST:-$SPLIT_ROOT/humaneval_train70_test30_seed${SPLIT_SEED}/split_manifest.json}"
MBPP_MANIFEST="${MBPP_MANIFEST:-$SPLIT_ROOT/mbpp_train70_test30_seed${SPLIT_SEED}/split_manifest.json}"

# One generation request contains up to 64 prompts. Each evaluation round
# contains 64 files and fans them out to 64 isolated evaluator containers.
BATCH_SIZE="${BATCH_SIZE:-64}"
PIPELINE_EVAL_BATCH_SIZE="${PIPELINE_EVAL_BATCH_SIZE:-64}"
EVAL_SHARDS="${EVAL_SHARDS:-64}"
EVAL_INNER_WORKERS="${EVAL_INNER_WORKERS:-1}"
PIPELINE_POLL_SECONDS="${PIPELINE_POLL_SECONDS:-0.5}"
COMPLETION_LIMIT="${COMPLETION_LIMIT:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINKING_MODE="${THINKING_MODE:-disabled}"
PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-qwen_chat_template_non_thinking_code_continuation_v4}"

DOCKER_EXEC="${DOCKER_EXEC:-docker}"
EVAL_IMAGE="${EVAL_IMAGE:-multipl-e-evaluation:jca-current}"
EVAL_QUIET="${EVAL_QUIET:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVALUATOR_PREFLIGHT="${SKIP_EVALUATOR_PREFLIGHT:-0}"

MODEL_TAG="${MODEL_TAG:-${MODEL_NAME//\//_}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ID="${RUN_ID:-${RUN_TIMESTAMP}_${MODEL_TAG}_8lang_sas_temp${TEMPERATURE//./p}}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/experiments/multipl_e_8lang_sas}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$MODEL_TAG/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-/data/wangyuheng/jca/logs/multipl_e_8lang_sas}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
COMPLETIONS_DIR="${COMPLETIONS_DIR:-$RUN_DIR/completions}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/run.log}"
CONFIG_PATH="${CONFIG_PATH:-$LOG_DIR/config.env}"
COMMAND_PATH="${COMMAND_PATH:-$LOG_DIR/command.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-$LOG_DIR/summary.txt}"
SCORES_PATH="${SCORES_PATH:-$LOG_DIR/sas_scores.json}"
STATUS_PATH="${STATUS_PATH:-$LOG_DIR/run_status.env}"
PIPELINE_METRICS_PATH="${PIPELINE_METRICS_PATH:-$LOG_DIR/pipeline_metrics.env}"
GENERATION_TIMINGS_PATH="${GENERATION_TIMINGS_PATH:-$LOG_DIR/generation_timings.jsonl}"
EVALUATION_TIMINGS_PATH="${EVALUATION_TIMINGS_PATH:-$LOG_DIR/evaluation_timings.jsonl}"
PREFLIGHT_PATH="${PREFLIGHT_PATH:-$LOG_DIR/evaluator_preflight.txt}"
MANIFEST_VALIDATION_PATH="${MANIFEST_VALIDATION_PATH:-$LOG_DIR/manifest_validation.json}"

for path_name in RUN_DIR LOG_DIR COMPLETIONS_DIR; do
    path_value="${!path_name}"
    if [[ "$path_value" != /* ]]; then
        printf -v "$path_name" '%s/%s' "$REPO_ROOT" "$path_value"
    fi
done

require_positive_uint() {
    local name="$1" value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer, got: $value" >&2
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

require_boolean() {
    local name="$1" value="$2"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "ERROR: $name must be 0 or 1, got: $value" >&2
        exit 2
    fi
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

for pair in \
    "NUM_GPUS:$NUM_GPUS" \
    "BATCH_SIZE:$BATCH_SIZE" \
    "PIPELINE_EVAL_BATCH_SIZE:$PIPELINE_EVAL_BATCH_SIZE" \
    "EVAL_SHARDS:$EVAL_SHARDS" \
    "EVAL_INNER_WORKERS:$EVAL_INNER_WORKERS" \
    "COMPLETION_LIMIT:$COMPLETION_LIMIT" \
    "MAX_TOKENS:$MAX_TOKENS"; do
    require_positive_uint "${pair%%:*}" "${pair#*:}"
done
require_boolean EVAL_QUIET "$EVAL_QUIET"
require_boolean DRY_RUN "$DRY_RUN"
require_boolean SKIP_EVALUATOR_PREFLIGHT "$SKIP_EVALUATOR_PREFLIGHT"

if [[ ! "$TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: TEMPERATURE must be a non-negative decimal, got: $TEMPERATURE" >&2
    exit 2
fi
if [[ "$THINKING_MODE" != "disabled" ]]; then
    echo "ERROR: THINKING_MODE must be disabled for this benchmark" >&2
    exit 2
fi
if [[ "$PROMPT_PROTOCOL" != "qwen_chat_template_non_thinking_code_continuation_v4" ]]; then
    echo "ERROR: unsupported PROMPT_PROTOCOL: $PROMPT_PROTOCOL" >&2
    exit 2
fi

EXPECTED_LANGUAGES="py,cpp,java,php,ts,cs,sh,js"
if [[ "${LANGUAGES// /}" != "$EXPECTED_LANGUAGES" ]]; then
    echo "ERROR: LANGUAGES must be exactly $EXPECTED_LANGUAGES" >&2
    exit 2
fi

require_file "SAS generator" "$REPO_ROOT/scripts/generate_multipl_e_sas.py"
require_file "pipeline scheduler" "$REPO_ROOT/scripts/run_multipl_e_pipeline.py"
require_file "parallel evaluator" "$REPO_ROOT/scripts/evaluate_multipl_e_parallel.py"
require_file "SAS summarizer" "$REPO_ROOT/scripts/summarize_multipl_e_sas.py"
require_file "evaluator preflight" "$REPO_ROOT/scripts/check_multipl_e_evaluator.py"
require_file "manifest validator" "$REPO_ROOT/scripts/validate_multipl_e_8lang_manifests.py"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
MANIFEST_PATHS=()
for dataset in "${DATASET_LIST[@]}"; do
    dataset="${dataset// /}"
    case "$dataset" in
        humaneval) manifest="$HUMANEVAL_MANIFEST" ;;
        mbpp) manifest="$MBPP_MANIFEST" ;;
        *)
            echo "ERROR: unsupported dataset '$dataset'; use humaneval and/or mbpp" >&2
            exit 2
            ;;
    esac
    require_file "$dataset split manifest" "$manifest"
    MANIFEST_PATHS+=("$manifest")
done
if [[ "${#MANIFEST_PATHS[@]}" -eq 0 ]]; then
    echo "ERROR: DATASETS selected no manifests" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$COMPLETIONS_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

VALIDATE_CMD=("$PYTHON_BIN" "$REPO_ROOT/scripts/validate_multipl_e_8lang_manifests.py" --seed "$SPLIT_SEED")
for manifest in "${MANIFEST_PATHS[@]}"; do
    VALIDATE_CMD+=(--manifest "$manifest")
done
"${VALIDATE_CMD[@]}" > "$MANIFEST_VALIDATION_PATH"

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: model directory does not exist: $MODEL_PATH" >&2
        exit 2
    fi
    if [[ -f "$MODEL_PATH/adapter_config.json" && ! -f "$MODEL_PATH/config.json" ]]; then
        echo "ERROR: $MODEL_PATH is a LoRA adapter-only checkpoint" >&2
        exit 2
    fi
    require_file "model config" "$MODEL_PATH/config.json"
fi

IFS=',' read -r -a LANGUAGE_LIST <<< "$LANGUAGES"
LANGUAGE_ARGS=()
for language in "${LANGUAGE_LIST[@]}"; do
    LANGUAGE_ARGS+=(--language "${language// /}")
done

GEN_CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_multipl_e_sas.py"
    --model-path "$MODEL_PATH"
    --split-manifest "${MANIFEST_PATHS[@]}"
    --output-dir "$COMPLETIONS_DIR"
    --num-gpus "$NUM_GPUS"
    --batch-size "$BATCH_SIZE"
    --completion-limit "$COMPLETION_LIMIT"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-tokens "$MAX_TOKENS"
    --thinking-mode "$THINKING_MODE"
    --prompt-protocol "$PROMPT_PROTOCOL"
    --timings-file "$GENERATION_TIMINGS_PATH"
    "${LANGUAGE_ARGS[@]}"
)

PIPELINE_CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_multipl_e_pipeline.py"
    --input-dir "$COMPLETIONS_DIR"
    --expected-completions "$COMPLETION_LIMIT"
    --eval-image "$EVAL_IMAGE"
    --docker-exec "$DOCKER_EXEC"
    --eval-shards "$EVAL_SHARDS"
    --eval-inner-workers "$EVAL_INNER_WORKERS"
    --eval-batch-size "$PIPELINE_EVAL_BATCH_SIZE"
    --poll-interval "$PIPELINE_POLL_SECONDS"
    --metrics-file "$PIPELINE_METRICS_PATH"
    --timings-file "$EVALUATION_TIMINGS_PATH"
    --evaluator-script "$REPO_ROOT/scripts/evaluate_multipl_e_parallel.py"
)
if [[ "$EVAL_QUIET" == "1" ]]; then
    PIPELINE_CMD+=(--quiet-evaluator)
fi
PIPELINE_CMD+=(-- "${GEN_CMD[@]}")

{
    printf 'RUN_NAME=multipl_e_8lang_sas\n'
    printf 'RUN_ID=%q\nSTARTED_AT=%q\n' "$RUN_ID" "$SCRIPT_STARTED_AT"
    printf 'REPO_ROOT=%q\nDATASET_ROOT=%q\n' "$REPO_ROOT" "$DATASET_ROOT"
    printf 'MODEL_NAME=%q\nMODEL_PATH=%q\nNUM_GPUS=%q\n' "$MODEL_NAME" "$MODEL_PATH" "$NUM_GPUS"
    printf 'DATASETS=%q\nLANGUAGES=%q\nSPLIT_SEED=%q\n' "$DATASETS" "$LANGUAGES" "$SPLIT_SEED"
    printf 'BATCH_SIZE=%q\nPIPELINE_EVAL_BATCH_SIZE=%q\nEVAL_SHARDS=%q\n' "$BATCH_SIZE" "$PIPELINE_EVAL_BATCH_SIZE" "$EVAL_SHARDS"
    printf 'EVAL_INNER_WORKERS=%q\nPIPELINE_POLL_SECONDS=%q\n' "$EVAL_INNER_WORKERS" "$PIPELINE_POLL_SECONDS"
    printf 'COMPLETION_LIMIT=%q\nTEMPERATURE=%q\nTOP_P=%q\nMAX_TOKENS=%q\n' "$COMPLETION_LIMIT" "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS"
    printf 'THINKING_MODE=%q\nPROMPT_PROTOCOL=%q\n' "$THINKING_MODE" "$PROMPT_PROTOCOL"
    printf 'EVAL_IMAGE=%q\nDOCKER_EXEC=%q\n' "$EVAL_IMAGE" "$DOCKER_EXEC"
    printf 'RUN_DIR=%q\nCOMPLETIONS_DIR=%q\nLOG_DIR=%q\n' "$RUN_DIR" "$COMPLETIONS_DIR" "$LOG_DIR"
    printf 'MANIFEST_VALIDATION_PATH=%q\n' "$MANIFEST_VALIDATION_PATH"
    for manifest in "${MANIFEST_PATHS[@]}"; do
        printf 'SPLIT_MANIFEST=%q\n' "$manifest"
    done
} > "$CONFIG_PATH"

{
    printf 'Manifest validation command:\n'
    print_command "${VALIDATE_CMD[@]}"
    printf 'Generator command:\n'
    print_command "${GEN_CMD[@]}"
    printf 'Pipeline command:\n'
    print_command "${PIPELINE_CMD[@]}"
} > "$COMMAND_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run passed; manifests validated and commands written to $COMMAND_PATH"
    cat "$COMMAND_PATH"
    exit 0
fi

if [[ "$SKIP_EVALUATOR_PREFLIGHT" != "1" ]]; then
    set +e
    "$PYTHON_BIN" "$REPO_ROOT/scripts/check_multipl_e_evaluator.py" \
        --image "$EVAL_IMAGE" --docker-exec "$DOCKER_EXEC" \
        2>&1 | tee "$PREFLIGHT_PATH"
    PREFLIGHT_STATUS=${PIPESTATUS[0]}
    set -e
    if [[ "$PREFLIGHT_STATUS" -ne 0 ]]; then
        exit "$PREFLIGHT_STATUS"
    fi
fi

STATUS=0
set +e
{
    echo "===== MultiPL-E eight-language SAS pipeline ====="
    echo "Started: $SCRIPT_STARTED_AT"
    echo "Model: $MODEL_NAME ($MODEL_PATH)"
    echo "Datasets: $DATASETS"
    echo "Languages: $LANGUAGES"
    echo "Split seed: $SPLIT_SEED"
    echo "Generation batch size: $BATCH_SIZE"
    echo "Evaluation batch/shards: $PIPELINE_EVAL_BATCH_SIZE/$EVAL_SHARDS"
    echo "Max output tokens: $MAX_TOKENS"
    echo "Completions: $COMPLETIONS_DIR"
    echo "Pipeline command:"
    print_command "${PIPELINE_CMD[@]}"
    echo
    "${PIPELINE_CMD[@]}"
} 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -eq 0 ]]; then
    SUMMARY_CMD=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_multipl_e_sas.py"
        --split-manifest "${MANIFEST_PATHS[@]}"
        --completions-dir "$COMPLETIONS_DIR"
        --output "$SCORES_PATH"
        --model-name "$MODEL_NAME"
        --model-path "$MODEL_PATH"
        --text-output "$SUMMARY_PATH"
    )
    for language in "${LANGUAGE_LIST[@]}"; do
        SUMMARY_CMD+=(--language "${language// /}")
    done
    set +e
    "${SUMMARY_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
    STATUS=${PIPESTATUS[0]}
    set -e
fi

GENERATION_SECONDS=0
EVALUATION_WORK_SECONDS=0
if [[ -f "$PIPELINE_METRICS_PATH" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            GENERATION_SECONDS) GENERATION_SECONDS="$value" ;;
            EVALUATION_WORK_SECONDS) EVALUATION_WORK_SECONDS="$value" ;;
        esac
    done < "$PIPELINE_METRICS_PATH"
fi
COMPLETED_AT="$(date '+%Y-%m-%d %H:%M:%S %Z')"
{
    printf 'STATUS=%s\n' "$STATUS"
    printf 'GENERATION_SECONDS=%s\n' "$GENERATION_SECONDS"
    printf 'EVALUATION_WORK_SECONDS=%s\n' "$EVALUATION_WORK_SECONDS"
    printf 'ELAPSED_SECONDS=%s\n' "$SECONDS"
    printf 'RUN_ID=%q\nMODEL_NAME=%q\nMODEL_PATH=%q\n' "$RUN_ID" "$MODEL_NAME" "$MODEL_PATH"
    printf 'STARTED_AT=%q\nCOMPLETED_AT=%q\n' "$SCRIPT_STARTED_AT" "$COMPLETED_AT"
    printf 'LOG_PATH=%q\nSCORES_PATH=%q\n' "$LOG_PATH" "$SCORES_PATH"
} > "$STATUS_PATH"

{
    echo
    echo "===== Run completion ====="
    echo "Status: $STATUS"
    echo "Generation seconds: $GENERATION_SECONDS"
    echo "Evaluation work seconds: $EVALUATION_WORK_SECONDS"
    echo "Total elapsed seconds: $SECONDS"
    echo "Scores: $SCORES_PATH"
    echo "Summary: $SUMMARY_PATH"
} | tee -a "$LOG_PATH"
exit "$STATUS"
