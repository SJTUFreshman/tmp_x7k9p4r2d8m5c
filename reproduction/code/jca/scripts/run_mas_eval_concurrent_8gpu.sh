#!/usr/bin/env bash
# Concurrent 8-GPU MAS evaluation using the old-SFT verification protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/run_sft_lora_vllm_8gpu.sh"

MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-old_sft}"
SPLIT="${SPLIT:-dev}"
START="${START:-0}"
LIMIT="${LIMIT:-2417}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-0}"
LOG_DIR="${LOG_DIR:-logs/mas_eval_concurrent}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mas_eval_concurrent}"

if [[ ! "$MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_CONCURRENCY must be a positive integer: $MAX_CONCURRENCY" >&2
  exit 1
fi

case "$EVAL_PROTOCOL" in
  old_sft|old|sft) ;;
  *)
    echo "ERROR: concurrent evaluator supports only the old-SFT protocol." >&2
    exit 1
    ;;
esac

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_mas_eval_c${MAX_CONCURRENCY}_${SPLIT}_start${START}_n${LIMIT}}"
EXTRA_ARGS="${EXTRA_ARGS:-} --max-concurrency ${MAX_CONCURRENCY}"

export EVAL_PROTOCOL SPLIT START LIMIT TEMPERATURE LOG_RAW_CHARS
export LOG_DIR OUTPUT_DIR RUN_ID EXTRA_ARGS

exec bash "$BASE_LAUNCHER" "$@"
