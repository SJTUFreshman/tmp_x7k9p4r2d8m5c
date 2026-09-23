#!/usr/bin/env bash
# JCA RL Training — RWR launcher.
#
# Trains 3 LoRA adapters (A1, A2, A3) sequentially using
# reward-weighted regression, starting from SFT checkpoints.
#
# Quick smoke test:
#   bash scripts/rl_train.sh
#
# Override adapters or rollout data:
#   ROLLOUT=/path/rollout.jsonl SFT_A1=/path/A1/final bash scripts/rl_train.sh

set -euo pipefail

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
ACCELERATE="${ACCELERATE:-accelerate}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

# ============================================================
# Rollout data
# ============================================================
ROLLOUT="${ROLLOUT:-rl_data/rollout_train_0_500.jsonl}"
REWARD_FIELD="${REWARD_FIELD:-reward}"
# Set explicitly for every run.  If left empty, rl_train.py can infer a
# homogeneous mode from rollout metadata, but legacy rows without metadata are
# rejected instead of silently changing the chat template.
TRAIN_ENABLE_THINKING="${TRAIN_ENABLE_THINKING:-}"

# ============================================================
# SFT adapter starting points
# ============================================================
SFT_A1="${SFT_A1:-/data/wangyuheng/jca/sft_runs/0706v2/A1/final}"
SFT_A2="${SFT_A2:-/data/wangyuheng/jca/sft_runs/0706v2/A2/final}"
SFT_A3="${SFT_A3:-/data/wangyuheng/jca/sft_runs/0706v2/A3/final}"

# ============================================================
# Base models
# ============================================================
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

# ============================================================
# RWR hyperparams
# ============================================================
KL_COEF="${KL_COEF:-0.05}"

# ============================================================
# Training hyperparams
# ============================================================
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-6}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ="${MAX_SEQ:-10000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ============================================================
# GPU / precision
# ============================================================
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# ============================================================
# Logging
# ============================================================
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RESUME="${RESUME:-0}"

# ============================================================
# Which agents to train
# ============================================================
AGENTS="${AGENTS:-A1 A2 A3}"

# ============================================================
# Output
# ============================================================
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)_rl}"
RL_RUNS_DIR="${RL_RUNS_DIR:-rl_runs/$TAG}"

# ============================================================
# Setup
# ============================================================
cd "$PROJECT_ROOT"
mkdir -p "$RL_RUNS_DIR"

LAUNCH_LOG="$RL_RUNS_DIR/launch.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "================================================================"
echo "JCA RL (RWR) launcher"
echo "================================================================"
echo "TAG           = $TAG"
echo "ROLLOUT       = $ROLLOUT"
echo "REWARD_FIELD  = $REWARD_FIELD"
echo "THINKING      = ${TRAIN_ENABLE_THINKING:-from-rollout-metadata}"
echo "AGENTS        = $AGENTS"
echo ""
echo "SFT_A1        = $SFT_A1"
echo "SFT_A2        = $SFT_A2"
echo "SFT_A3        = $SFT_A3"
echo ""
echo "MODEL_A1      = $MODEL_A1"
echo "MODEL_A2      = $MODEL_A2"
echo "MODEL_A3      = $MODEL_A3"
echo ""
echo "KL_COEF       = $KL_COEF"
echo "NUM_EPOCHS    = $NUM_EPOCHS"
echo "LR            = $LR"
echo "SEED          = $SEED"
echo "PER_DEV_BATCH = $PER_DEVICE_BATCH"
echo "GRAD_ACCUM    = $GRAD_ACCUM"
echo "MAX_SEQ       = $MAX_SEQ"
echo "LORA_RANK     = $LORA_RANK"
echo "LORA_ALPHA    = $LORA_ALPHA"
echo "NUM_GPUS      = $NUM_GPUS"
echo "MIXED_PREC    = $MIXED_PRECISION"
echo "RESUME        = $RESUME"
echo "RL_RUNS_DIR   = $RL_RUNS_DIR"
echo "================================================================"
echo ""

# Preflight
[[ -f "$ROLLOUT" ]] || { echo "[fatal] rollout file not found: $ROLLOUT"; exit 1; }
[[ -f "scripts/rl_train.py" ]] || { echo "[fatal] scripts/rl_train.py not found under $PROJECT_ROOT"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "[fatal] SEED must be a non-negative integer"; exit 1; }
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || { echo "[fatal] RESUME must be 0 or 1"; exit 1; }

# ============================================================
# Train each agent
# ============================================================
for AGENT in $AGENTS; do
    MODEL_VAR="MODEL_$AGENT"
    SFT_VAR="SFT_$AGENT"
    MODEL_PATH="${!MODEL_VAR}"
    SFT_PATH="${!SFT_VAR}"
    AGENT_OUT="$RL_RUNS_DIR/$AGENT"

    [[ -d "$MODEL_PATH" ]] || { echo "[skip] $AGENT: model not found: $MODEL_PATH"; continue; }
    [[ -d "$SFT_PATH"   ]] || { echo "[skip] $AGENT: SFT adapter not found: $SFT_PATH"; continue; }
    [[ -f "$SFT_PATH/adapter_model.safetensors" ]] || {
        echo "[skip] $AGENT: adapter weights not found in $SFT_PATH"; continue
    }

    echo "================================================================"
    echo "[RL] Training $AGENT"
    echo "  base:        $MODEL_PATH"
    echo "  sft adapter: $SFT_PATH"
    echo "  rollout:     $ROLLOUT"
    echo "  out:         $AGENT_OUT"
    echo "================================================================"
    mkdir -p "$AGENT_OUT"

    LAUNCH_ARGS=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
    if [[ "$NUM_GPUS" != "1" ]]; then
        LAUNCH_ARGS+=(--multi_gpu)
    fi

    TRAIN_ARGS=(
        --agent              "$AGENT" \
        --rollout            "$ROLLOUT" \
        --reward-field       "$REWARD_FIELD" \
        --sft-adapter        "$SFT_PATH" \
        --out-dir            "$AGENT_OUT" \
        --model-name-or-path "$MODEL_PATH" \
        --kl-coef            "$KL_COEF" \
        --num-epochs         "$NUM_EPOCHS" \
        --learning-rate      "$LR" \
        --seed               "$SEED" \
        --per-device-batch-size "$PER_DEVICE_BATCH" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --max-seq-length     "$MAX_SEQ" \
        --lora-rank          "$LORA_RANK" \
        --lora-alpha         "$LORA_ALPHA" \
        --lora-dropout       "$LORA_DROPOUT" \
        --warmup-ratio       "$WARMUP_RATIO" \
        --logging-steps      "$LOGGING_STEPS" \
        --save-steps         "$SAVE_STEPS" \
        --save-total-limit   "$SAVE_TOTAL_LIMIT" \
        --report-to          "$REPORT_TO" \
        --gradient-checkpointing \
        --bf16
    )
    if [[ -n "$TRAIN_ENABLE_THINKING" ]]; then
        if [[ "$TRAIN_ENABLE_THINKING" == "1" ]]; then
            TRAIN_ARGS+=(--enable-thinking)
        elif [[ "$TRAIN_ENABLE_THINKING" == "0" ]]; then
            TRAIN_ARGS+=(--no-enable-thinking)
        else
            echo "[fatal] TRAIN_ENABLE_THINKING must be 0, 1, or empty" >&2
            exit 1
        fi
    fi
    if [[ "$RESUME" == "1" ]]; then
        TRAIN_ARGS+=(--resume-from-checkpoint latest)
    fi

    PYTHONPATH="$PYTHONPATH_ROOT" "$ACCELERATE" launch \
        "${LAUNCH_ARGS[@]}" \
        scripts/rl_train.py \
        "${TRAIN_ARGS[@]}"

    echo "[done] $AGENT — final adapter at $AGENT_OUT/final"
    echo ""
done

echo "================================================================"
echo "All requested agents trained."
echo "Final adapters: $RL_RUNS_DIR/{A1,A2,A3}/final"
echo "TensorBoard:    tensorboard --logdir $RL_RUNS_DIR"
echo "================================================================"
