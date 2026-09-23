#!/usr/bin/env bash
# JCA SFT — one-button launcher.
#
# Trains 3 LoRA adapters (A1, A2, A3) sequentially on the same node,
# each using all 8 GPUs via `accelerate launch --num_processes 8 --multi_gpu`.
#
# Sequential rather than parallel because:
#   - Each agent needs all 8 GPUs to fit Qwen3-8B comfortably.
#   - Running them in parallel on disjoint GPU sets is harder to manage and
#     makes per-agent comparison messy in shared logs.
#
# Inputs:
#   $TRAJ_FILE  trajectories.jsonl from self-distillation (sanity or full).
#               MULTIPLE files can be passed separated by COMMA — they'll be
#               concatenated before filtering (useful for merging 2-hop +
#               3/4-hop sanity runs).
#   $TAG        run tag, used for output directory naming
#
# Outputs:
#   jca/sft_data/$TAG/{A1,A2,A3}.jsonl   per-agent SFT data
#   jca/sft_data/$TAG/stats.json
#   jca/sft_runs/$TAG/$AGENT/            checkpoints + logs per agent
#   jca/sft_runs/$TAG/$AGENT/final/      best LoRA adapter (load_best_model_at_end)
#   jca/sft_runs/$TAG/launch.log         this launcher's stdout
#
# Usage:
#   bash jca/scripts/sft_train.sh outputs/self_distill_sanity/<run>/trajectories.jsonl <tag>
#
# If SFT data is already built:
#   PREBUILT_SFT_DATA_DIR=/data/wangyuheng/jca/v1 bash jca/scripts/sft_train.sh v1
#
# Optional env vars to override defaults:
#   PREBUILT_SFT_DATA_DIR, PROJ_ROOT, SPLIT, MUSIQUE_DIR, EVAL_FRACTION, NUM_EPOCHS, LR, MAX_PER_AGENT,
#   PER_DEVICE_BATCH, GRAD_ACCUM, MAX_SEQ, REPORT_TO, EARLY_STOP_PATIENCE,
#   AGENTS (default "A1 A2 A3", set to a subset to skip), MODEL_A1, MODEL_A2, MODEL_A3,
#   PER_AGENT_DATA (1 uses A1.jsonl/A2.jsonl/A3.jsonl), SFT_SPLIT_GROUP_BY

set -euo pipefail

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
PREBUILT_SFT_DATA_DIR="${PREBUILT_SFT_DATA_DIR:-}"

if [ -n "$PREBUILT_SFT_DATA_DIR" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Usage with prebuilt data: PREBUILT_SFT_DATA_DIR=/path/to/sft_data $0 <tag>"
        echo "Example: PREBUILT_SFT_DATA_DIR=/data/wangyuheng/jca/v1 $0 v1"
        exit 1
    fi
    TRAJ_FILE=""
    TAG="$1"
    TRAJ_FILES=()
else
    if [ "$#" -lt 2 ]; then
        echo "Usage: $0 <trajectories.jsonl> <tag>"
        echo "Example: $0 outputs/self_distill_sanity/20260630_171827_n100k8/trajectories.jsonl v0"
        echo "Prebuilt example: PREBUILT_SFT_DATA_DIR=/data/wangyuheng/jca/v1 $0 v1"
        exit 1
    fi

    TRAJ_FILE="$1"
    TAG="$2"

    # Split TRAJ_FILE on comma into an array; verify each path exists
    IFS=',' read -ra TRAJ_FILES <<< "$TRAJ_FILE"
    for tf in "${TRAJ_FILES[@]}"; do
        if [ ! -f "$tf" ]; then
            echo "[fatal] trajectories file not found: $tf"
            exit 1
        fi
    done
fi

# -----------------------------------------------------------------------------
# Defaults (override via env)
# -----------------------------------------------------------------------------
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SPLIT="${SPLIT:-train}"
MUSIQUE_DIR="${MUSIQUE_DIR:-jca/musique_data}"

# SFT data builder
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
MAX_PER_AGENT="${MAX_PER_AGENT:-}"   # empty = no cap

# Trainer
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-2e-5}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-4096}"
MAX_TOKENIZATION_DROP_RATIO="${MAX_TOKENIZATION_DROP_RATIO:-0.05}"
REPORT_TO="${REPORT_TO:-tensorboard}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
NUM_GEN_SAMPLES="${NUM_GEN_SAMPLES:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-0}"
PER_AGENT_DATA="${PER_AGENT_DATA:-0}"
SFT_SPLIT_GROUP_BY="${SFT_SPLIT_GROUP_BY:-problem_id}"

if [ "$PER_AGENT_DATA" != "0" ] && [ "$PER_AGENT_DATA" != "1" ]; then
    echo "[fatal] PER_AGENT_DATA must be 0 or 1, got: $PER_AGENT_DATA"
    exit 1
fi

# GPU launcher
NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE="${ACCELERATE:-accelerate}"
PY="${PY:-python}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
if [ "$MIXED_PRECISION" != "bf16" ] && [ "$MIXED_PRECISION" != "fp16" ] && [ "$MIXED_PRECISION" != "no" ]; then
    echo "[fatal] MIXED_PRECISION must be bf16, fp16, or no; got: $MIXED_PRECISION"
    exit 1
fi

# Which agents to train (allow skipping)
AGENTS="${AGENTS:-A1 A2 A3}"

# Local model defaults. Override if you want HF IDs or another checkpoint.
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

# Optional: init LoRA from existing adapter instead of creating from scratch.
# Set to a directory containing adapter_model.safetensors to continue training
# from a previous LoRA (e.g. RL checkpoint). Leave empty for fresh LoRA.
INIT_LORA_A1="${INIT_LORA_A1:-}"
INIT_LORA_A2="${INIT_LORA_A2:-}"
INIT_LORA_A3="${INIT_LORA_A3:-}"

# -----------------------------------------------------------------------------
# Setup output dirs
# -----------------------------------------------------------------------------
cd "$PROJ_ROOT"

if [ -n "$PREBUILT_SFT_DATA_DIR" ]; then
    SFT_DATA_DIR="$PREBUILT_SFT_DATA_DIR"
else
    SFT_DATA_DIR="jca/sft_data/$TAG"
fi
SFT_RUNS_DIR="jca/sft_runs/$TAG"
if [ -z "$PREBUILT_SFT_DATA_DIR" ]; then
    mkdir -p "$SFT_DATA_DIR"
fi
mkdir -p "$SFT_RUNS_DIR"

LAUNCH_LOG="$SFT_RUNS_DIR/launch.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "================================================================"
echo "JCA SFT launcher"
echo "================================================================"
echo "PROJ_ROOT     = $PROJ_ROOT"
echo "TRAJ_FILE     = ${TRAJ_FILES[*]:-<prebuilt>}"
echo "TAG           = $TAG"
echo "PREBUILT_DATA = ${PREBUILT_SFT_DATA_DIR:-<none>}"
echo "SFT_DATA_DIR  = $SFT_DATA_DIR"
echo "SFT_RUNS_DIR  = $SFT_RUNS_DIR"
echo "AGENTS        = $AGENTS"
echo "MODEL_A1      = $MODEL_A1"
echo "MODEL_A2      = $MODEL_A2"
echo "MODEL_A3      = $MODEL_A3"
echo "NUM_GPUS      = $NUM_GPUS"
echo "MIXED_PREC    = $MIXED_PRECISION"
echo "NUM_EPOCHS    = $NUM_EPOCHS"
echo "LR            = $LR"
echo "PER_DEV_BATCH = $PER_DEVICE_BATCH"
echo "GRAD_ACCUM    = $GRAD_ACCUM"
echo "MAX_SEQ       = $MAX_SEQ"
echo "MAX_TOK_DROP  = $MAX_TOKENIZATION_DROP_RATIO"
echo "LORA_RANK     = $LORA_RANK"
echo "LORA_ALPHA    = $LORA_ALPHA"
echo "LORA_DROPOUT  = $LORA_DROPOUT"
echo "LOAD_BEST     = $LOAD_BEST_MODEL_AT_END"
echo "EVAL_FRACTION = $EVAL_FRACTION"
echo "PER_AGENT_DATA= $PER_AGENT_DATA"
echo "SPLIT_GROUP_BY= $SFT_SPLIT_GROUP_BY"
echo "REPORT_TO     = $REPORT_TO"
echo "EARLY_STOP    = $EARLY_STOP_PATIENCE"
echo "DL_WORKERS    = $DATALOADER_NUM_WORKERS"
echo "================================================================"
echo

echo "[preflight] Checking Python dependencies..."
"$PY" - "$REPORT_TO" <<'PY'
import importlib.util
import sys

report_to = sys.argv[1]
required = ["torch", "transformers", "accelerate", "datasets", "peft"]
if report_to in {"tensorboard", "all"} or "tensorboard" in report_to.split(","):
    required.append("tensorboard")

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("[fatal] missing Python package(s): " + ", ".join(missing))
    print("[hint] install in the same environment used to launch training, e.g.:")
    print("       python -m pip install -U " + " ".join(missing))
    raise SystemExit(1)
print("[preflight] OK: " + ", ".join(required))
PY
echo

# -----------------------------------------------------------------------------
# Phase 1: Build per-agent SFT data
# -----------------------------------------------------------------------------
if [ -n "$PREBUILT_SFT_DATA_DIR" ]; then
    echo "[phase 1] Using prebuilt SFT data, skipping data build."
    if [ ! -f "$SFT_DATA_DIR/stats.json" ]; then
        echo "[fatal] prebuilt stats.json not found: $SFT_DATA_DIR/stats.json"
        exit 1
    fi
    if [ ! -f "$SFT_DATA_DIR/all_agents.jsonl" ]; then
        echo "[fatal] prebuilt all_agents.jsonl not found: $SFT_DATA_DIR/all_agents.jsonl"
        exit 1
    fi
    cat "$SFT_DATA_DIR/stats.json"
    echo
elif [ ! -f "$SFT_DATA_DIR/stats.json" ]; then
    echo "[phase 1] Building per-agent SFT data..."
    BUILD_ARGS=(
        --input "${TRAJ_FILES[@]}"
        --out-dir "$SFT_DATA_DIR"
        --data-dir "$MUSIQUE_DIR"
        --split "$SPLIT"
        --require-non-forced
        --require-correct
    )
    if [ -n "$MAX_PER_AGENT" ]; then
        BUILD_ARGS+=(--max-per-agent "$MAX_PER_AGENT")
    fi
    "$PY" jca/scripts/build_sft_data.py "${BUILD_ARGS[@]}"
    echo
else
    echo "[phase 1] SFT data already built ($SFT_DATA_DIR/stats.json exists), skipping."
    cat "$SFT_DATA_DIR/stats.json"
    echo
fi

# -----------------------------------------------------------------------------
# Phase 2: Train each agent
# -----------------------------------------------------------------------------
for AGENT in $AGENTS; do
    if [ "$PER_AGENT_DATA" = "1" ]; then
        DATA_FILE="$SFT_DATA_DIR/$AGENT.jsonl"
    else
        DATA_FILE="$SFT_DATA_DIR/all_agents.jsonl"
    fi
    AGENT_OUT="$SFT_RUNS_DIR/$AGENT"
    MODEL_PATH_VAR="MODEL_$AGENT"
    MODEL_PATH="${!MODEL_PATH_VAR}"

    if [ ! -f "$DATA_FILE" ]; then
        echo "[skip] $AGENT: no data file at $DATA_FILE"
        continue
    fi
    if [ ! -e "$MODEL_PATH" ]; then
        echo "[skip] $AGENT: model path does not exist: $MODEL_PATH"
        continue
    fi

    N_EX=$(wc -l < "$DATA_FILE" | tr -d ' ')
    if [ "$N_EX" -lt 16 ]; then
        echo "[skip] $AGENT: too few examples ($N_EX)"
        continue
    fi

    echo "================================================================"
    echo "[phase 2] Training $AGENT — $N_EX examples"
    echo "  data: $DATA_FILE"
    echo "  base: $MODEL_PATH"
    echo "  out:  $AGENT_OUT"
    echo "================================================================"
    mkdir -p "$AGENT_OUT"

    LOAD_BEST_ARG="--no-load-best-model-at-end"
    if [ "$LOAD_BEST_MODEL_AT_END" = "1" ]; then
        LOAD_BEST_ARG="--load-best-model-at-end"
    fi

    INIT_LORA_VAR="INIT_LORA_$AGENT"
    INIT_LORA_PATH="${!INIT_LORA_VAR}"
    INIT_LORA_ARGS=()
    if [ -n "$INIT_LORA_PATH" ] && [ -d "$INIT_LORA_PATH" ]; then
        INIT_LORA_ARGS=(--init-lora-adapter "$INIT_LORA_PATH")
        echo "  init LoRA from: $INIT_LORA_PATH"
    fi

    PRECISION_ARGS=(--bf16)
    if [ "$MIXED_PRECISION" = "fp16" ]; then
        PRECISION_ARGS=(--no-bf16 --fp16)
    elif [ "$MIXED_PRECISION" = "no" ]; then
        PRECISION_ARGS=(--no-bf16)
    fi

    $ACCELERATE launch \
        --num_processes "$NUM_GPUS" \
        --multi_gpu \
        --mixed_precision "$MIXED_PRECISION" \
        jca/scripts/sft_train.py \
        --agent "$AGENT" \
        --data "$DATA_FILE" \
        --out-dir "$AGENT_OUT" \
        --model-name-or-path "$MODEL_PATH" \
        --eval-fraction "$EVAL_FRACTION" \
        --split-group-by "$SFT_SPLIT_GROUP_BY" \
        --num-epochs "$NUM_EPOCHS" \
        --learning-rate "$LR" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --per-device-train-batch-size "$PER_DEVICE_BATCH" \
        --per-device-eval-batch-size "$PER_DEVICE_BATCH" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --max-seq-length "$MAX_SEQ" \
        --max-tokenization-drop-ratio "$MAX_TOKENIZATION_DROP_RATIO" \
        --logging-steps "$LOGGING_STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --early-stopping-patience "$EARLY_STOP_PATIENCE" \
        "$LOAD_BEST_ARG" \
        --num-eval-samples-to-generate "$NUM_GEN_SAMPLES" \
        --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
        --report-to "$REPORT_TO" \
        --gradient-checkpointing \
        "${PRECISION_ARGS[@]}" \
        "${INIT_LORA_ARGS[@]}"

    echo "[done] $AGENT trained — final adapter at $AGENT_OUT/final"
    echo
done

echo "================================================================"
echo "All requested agents trained."
echo "Run logs:    $SFT_RUNS_DIR/*/train_*.log"
echo "TensorBoard: tensorboard --logdir $SFT_RUNS_DIR"
echo "Final adapters: $SFT_RUNS_DIR/{A1,A2,A3}/final"
echo "================================================================"
