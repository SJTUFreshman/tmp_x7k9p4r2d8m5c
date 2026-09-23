#!/usr/bin/env bash
# v13 14B-scored role-specific c2c=(1,.5,1), trained from the original SFT
# adapters so this remains an isolated reward ablation.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${PROJECT_ROOT}/scripts/gsm_judge_rl/run_pipeline.sh}"
SFT_TAG="${SFT_TAG:-gsm_fixed_a1_corr30_20260723_v3}"
SOURCE_INPUT="${V13_14B_SOURCE_INPUT:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/v13_14b/source_rejudged.jsonl}"
VERSION_TAG="${VERSION_TAG:-gsm_judge_rl_v13_14b_role_c2c_1_05_1}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/${VERSION_TAG}}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/rl_runs/gsm_judge_rl/${VERSION_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/gsm_eval/gsm_judge_rl/${VERSION_TAG}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${PROJECT_ROOT}/logs/gsm_judge_rl/${VERSION_TAG}}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${PROJECT_ROOT}/logs/gsm_eval/gsm_judge_rl/${VERSION_TAG}}"
exec env \
  PROJECT_ROOT="$PROJECT_ROOT" TAG="$VERSION_TAG" SFT_TAG="$SFT_TAG" \
  SOURCE_INPUT="$SOURCE_INPUT" TRAIN_DATA="$DATA_DIR/train.jsonl" \
  HOLDOUT_DATA="$DATA_DIR/holdout.jsonl" DATA_STATS="$DATA_DIR/stats.json" \
  RL_RUNS_DIR="$RUN_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
  BASELINE_OUTPUT="$OUTPUT_DIR/sft_baseline_dev132.jsonl" \
  A1_ONLY_OUTPUT="$OUTPUT_DIR/a1_only_dev132.jsonl" \
  A2_ONLY_OUTPUT="$OUTPUT_DIR/a2_only_dev132.jsonl" \
  A3_ONLY_OUTPUT="$OUTPUT_DIR/a3_only_dev132.jsonl" \
  ALL_AGENTS_OUTPUT="$OUTPUT_DIR/all_agents_dev132.jsonl" \
  REPORT_OUTPUT="$OUTPUT_DIR/agent_ablation_report.json" \
  RUN_LOG_DIR="$TRAIN_LOG_DIR" EVAL_LOG_ROOT="$EVAL_LOG_DIR" \
  CORRECT_TO_CORRECT_COEF=1.0 CORRECT_TO_CORRECT_COEF_A1=1.0 \
  CORRECT_TO_CORRECT_COEF_A2=0.5 CORRECT_TO_CORRECT_COEF_A3=1.0 \
  CORRECT_TO_CORRECT_HANDOFF_COEF=1.0 TRANSITION_BOOST=1.25 \
  REWARD_FIELD=reward \
  bash "$PIPELINE_SCRIPT"
