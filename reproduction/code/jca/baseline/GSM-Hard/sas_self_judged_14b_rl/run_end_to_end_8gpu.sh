#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
RUN_ID="${RUN_ID:-gsm_sas14b_self_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl/runs/$RUN_ID}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT_ROOT/rl_runs/gsm_sas_self_rl/$RUN_ID}"
SCRIPT_DIR="$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl"

export PROJECT_ROOT RUN_ID RUN_DIR TRAIN_OUT

echo "========== GSM-HARD SAS SELF-RL =========="
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_OUT=$TRAIN_OUT"

bash "$SCRIPT_DIR/run_data_8gpu.sh"
bash "$SCRIPT_DIR/run_train_8gpu.sh"
ADAPTER="$TRAIN_OUT/final" OUTPUT="$RUN_DIR/sas_dev_eval.jsonl" \
  SUMMARY="$RUN_DIR/eval_summary.json" bash "$SCRIPT_DIR/run_eval_8gpu.sh"

echo "========== ALL DONE =========="
echo "judged_data=$RUN_DIR/sas_train_judged.jsonl"
echo "adapter=$TRAIN_OUT/final"
echo "eval=$RUN_DIR/sas_dev_eval.jsonl"
echo "summary=$RUN_DIR/eval_summary.json"
