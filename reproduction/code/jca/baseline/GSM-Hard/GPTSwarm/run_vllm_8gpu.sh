#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
SPLIT="${SPLIT:-dev}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ "$SPLIT" == "train" ]]; then
    DATA_PATH="$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl"
  else
    DATA_PATH="$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl"
  fi
fi

[[ "$ENABLE_THINKING" == 1 ]] || {
  echo "[fatal] GSM-Hard GPTSwarm must run with ENABLE_THINKING=1" >&2
  exit 1
}

export PROJECT_ROOT SPLIT ENABLE_THINKING
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export DATA_DIR="$DATA_PATH"
export LIMIT="${LIMIT:-132}"
export RUNNER_PATH="${RUNNER_PATH:-baseline/GSM-Hard/GPTSwarm/run_gptswarm_gsmhard.py}"
export NODES_PATH="${NODES_PATH:-baseline/MuSiQue/GPTSwarm/nodes.py}"
export SWARM_PATH="${SWARM_PATH:-baseline/MuSiQue/GPTSwarm/swarm.py}"
export PROMPT_IO_PATH="${PROMPT_IO_PATH:-baseline/GSM-Hard/GPTSwarm/prompts/node_io.md}"
export PROMPT_COT_PATH="${PROMPT_COT_PATH:-baseline/GSM-Hard/GPTSwarm/prompts/node_cot.md}"
export PROMPT_DEBATE_PATH="${PROMPT_DEBATE_PATH:-baseline/GSM-Hard/GPTSwarm/prompts/node_debate.md}"
export PROMPT_AGGREGATOR_PATH="${PROMPT_AGGREGATOR_PATH:-baseline/GSM-Hard/GPTSwarm/prompts/node_aggregator.md}"
export OUTPUT_DIR="${OUTPUT_DIR:-baseline/GSM-Hard/GPTSwarm/outputs}"
export LOG_DIR="${LOG_DIR:-baseline/GSM-Hard/GPTSwarm/logs}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN=1: no vLLM server or runner will be started"
  echo "  baseline=GPTSwarm split=$SPLIT data=$DATA_DIR limit=$LIMIT thinking=$ENABLE_THINKING"
  echo "  max_new_tokens=$MAX_NEW_TOKENS max_model_len=$MAX_MODEL_LEN"
  echo "  runner=$RUNNER_PATH"
  exit 0
fi

exec bash "$PROJECT_ROOT/baseline/MuSiQue/GPTSwarm/run_vllm_8gpu.sh"
