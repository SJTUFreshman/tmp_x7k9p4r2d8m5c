#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
TAG="${TAG:-multipl_e_a2_gpt5_paired_rwr_20260827}"
TRAIN_ROOT="${TRAIN_ROOT:-$PROJECT_ROOT/rl_runs/$TAG}"

exec env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}" \
    ROLLOUT="${ROLLOUT:-$PROJECT_ROOT/rl_data/clean/multipl_e_8lang_best_sft_gpt5_a2_paired_core_20260827.jsonl}" \
    EXPECTED_REWARD_SEMANTICS=per_turn_execution_v2 \
    REWARD_FIELD=train_weight \
    ALLOW_JUDGE_FAILED=0 \
    SFT_A2="${SFT_A2:-$PROJECT_ROOT/sft_runs/multipl_e_8lang_v5_diverse_a2_selfcorr2x_2ep_20260819/A2/final}" \
    AGENTS=A2 \
    RL_RUNS_DIR="$TRAIN_ROOT" \
    TAG="$TAG" \
    NUM_EPOCHS=1 \
    LR=5e-8 \
    KL_COEF=0.2 \
    PER_DEVICE_BATCH=1 \
    GRAD_ACCUM=2 \
    MAX_SEQ=12000 \
    WARMUP_RATIO=0.05 \
    LORA_RANK=64 \
    LORA_ALPHA=64 \
    LORA_DROPOUT=0.05 \
    SAVE_STEPS=20 \
    SAVE_TOTAL_LIMIT=10 \
    LOGGING_STEPS=5 \
    NUM_GPUS=8 \
    MIXED_PRECISION=bf16 \
    REPORT_TO=tensorboard \
    bash "$PROJECT_ROOT/scripts/rl_train_multipl_e.sh"
