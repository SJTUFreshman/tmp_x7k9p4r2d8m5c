#!/usr/bin/env bash
# JCA RL Training — RWR launcher (MultiPL-E code data, alpha=0.3).
#
# This is a sibling of rl_train.sh; only differences vs. the MuSiQue launcher:
#   - ROLLOUT points at the per-turn-execution rejudged MultiPL-E file
#   - SFT_A1/A2 default to the best score-v2 MultiPL-E SFT run
#   - MAX_SEQ bumped a bit — code trajectories can carry SOURCE PREFIX + several
#     candidate blocks per turn
#   - LR nudged down since the dataset is smaller (745 groups / 1914 rows)
#
# rl_train.py itself is UNCHANGED — it consumes the standard fields
# (problem_id, turn, agent_id, messages, response, reward) which our
# MultiPL-E data already provides.
#
# Quick run:
#   SFT_A1=/path/multipl_e_sft/A1/final \
#   SFT_A2=/path/multipl_e_sft/A2/final \
#   bash scripts/rl_train_multipl_e.sh

set -euo pipefail

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"

# ============================================================
# Rollout data — alpha=0.3 judged MultiPL-E
# ============================================================
ROLLOUT="${ROLLOUT:-rl_data/clean/multipl_e_rl_a1a2_sft_a3base_0811_judged_qwen14b_perturn_alpha03_clean.jsonl}"
REWARD_FIELD="${REWARD_FIELD:-reward}"
EXPECTED_REWARD_SEMANTICS="${EXPECTED_REWARD_SEMANTICS:-per_turn_execution_v2}"
ALLOW_JUDGE_FAILED="${ALLOW_JUDGE_FAILED:-0}"

# ============================================================
# SFT adapter starting points — MUST point at MultiPL-E SFT (not MuSiQue).
# Override these when launching if the paths differ on your box.
# ============================================================
SFT_ROOT="${SFT_ROOT:-/data/wangyuheng/jca/sft_runs/20260726_211500_multipl_e_8lang_3x8b_score_v2-1ep}"
SFT_A1="${SFT_A1:-$SFT_ROOT/A1/final}"
SFT_A2="${SFT_A2:-$SFT_ROOT/A2/final}"
SFT_A3="${SFT_A3:-$SFT_ROOT/A3/final}"
# Space-separated agents that should start from a fresh LoRA on the base
# model.  This is intentionally opt-in so existing launchers remain SFT-backed
# by default; it is used by the A1-only w/o-SFT control.
FRESH_LORA_AGENTS="${FRESH_LORA_AGENTS:-}"

# ============================================================
# Base models — same capacity ladder as MuSiQue setup
# ============================================================
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

# ============================================================
# RWR hyperparams
# ============================================================
# α used to build ROLLOUT is 0.3 — recorded here for the launch.log,
# not consumed by rl_train.py (reward is already baked into the file).
REWARD_ALPHA_NOTE="${REWARD_ALPHA_NOTE:-0.3}"
KL_COEF="${KL_COEF:-0.05}"

# ============================================================
# Training hyperparams
# ============================================================
# Notes vs rl_train.sh:
#   - LR: 3e-6 instead of 5e-6. Dataset is ~1914 rows (small);
#     lower LR reduces the chance of blowing up A2/A3 mid-run.
#   - MAX_SEQ: 12000 instead of 10000. Code turns render as
#     SOURCE PREFIX + prior candidates + reasoning + tentative_completion
#     which can grow beyond 10k on long sh/java trajectories.
#   - PER_DEVICE_BATCH / GRAD_ACCUM stay conservative to keep memory
#     safe with the larger sequence length.
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-3e-6}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ="${MAX_SEQ:-12000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ============================================================
# GPU / precision
# ============================================================
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# ============================================================
# Logging
# ============================================================
# Smaller dataset → fewer optimizer steps → log/save more often.
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-tensorboard}"

# ============================================================
# Which agents to train
# ============================================================
AGENTS="${AGENTS:-A1 A2}"

# ============================================================
# Output
# ============================================================
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)_rl_multipl_e_a03}"
RL_RUNS_DIR="${RL_RUNS_DIR:-rl_runs/$TAG}"

# ============================================================
# Setup
# ============================================================
cd "$PROJECT_ROOT"
mkdir -p "$RL_RUNS_DIR"

LAUNCH_LOG="$RL_RUNS_DIR/launch.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "================================================================"
echo "JCA RL (RWR) launcher — MultiPL-E α=$REWARD_ALPHA_NOTE"
echo "================================================================"
echo "TAG           = $TAG"
echo "ROLLOUT       = $ROLLOUT"
echo "REWARD_FIELD  = $REWARD_FIELD"
echo "REWARD_SCHEMA = $EXPECTED_REWARD_SEMANTICS"
echo "ALLOW_FAILED  = $ALLOW_JUDGE_FAILED"
echo "REWARD_ALPHA  = $REWARD_ALPHA_NOTE  (baked into ROLLOUT.reward)"
echo "AGENTS        = $AGENTS"
echo ""
echo "SFT_A1        = $SFT_A1"
echo "SFT_A2        = $SFT_A2"
echo "SFT_A3        = $SFT_A3"
echo "FRESH_LORA    = ${FRESH_LORA_AGENTS:-<none>}"
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
echo "RL_RUNS_DIR   = $RL_RUNS_DIR"
echo "================================================================"
echo ""

# Preflight
[[ -f "$ROLLOUT" ]] || { echo "[fatal] rollout file not found: $ROLLOUT"; exit 1; }
[[ -f "scripts/rl_train.py" ]] || { echo "[fatal] scripts/rl_train.py not found under $PROJECT_ROOT"; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] python not executable: $PYTHON_BIN"; exit 1; }
[[ -x "$ACCELERATE" ]] || { echo "[fatal] accelerate not executable: $ACCELERATE"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "[fatal] SEED must be a non-negative integer"; exit 1; }
[[ "$ALLOW_JUDGE_FAILED" == "0" || "$ALLOW_JUDGE_FAILED" == "1" ]] || {
    echo "[fatal] ALLOW_JUDGE_FAILED must be 0 or 1"; exit 1;
}

# Validate the complete file. This deliberately rejects legacy judged files
# whose task_reward was copied from the first turn across the trajectory.
"$PYTHON_BIN" - "$ROLLOUT" "$REWARD_FIELD" "$EXPECTED_REWARD_SEMANTICS" "$ALLOW_JUDGE_FAILED" <<'PY'
import collections
import json
import math
import sys

path, reward_field, expected_semantics, allow_judge_failed = sys.argv[1:]
allow_judge_failed = allow_judge_failed == "1"
rows = 0
groups = set()
agents = collections.Counter()
failed = 0

with open(path, encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[fatal] invalid JSON at {path}:{line_no}: {exc}")

        rows += 1
        if row.get("reward_semantics") != expected_semantics:
            raise SystemExit(
                f"[fatal] stale reward semantics at {path}:{line_no}: "
                f"{row.get('reward_semantics')!r}, expected {expected_semantics!r}"
            )
        try:
            reward = float(row[reward_field])
            task_reward = float(row["task_reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"[fatal] invalid reward fields at {path}:{line_no}: {exc}")
        if not math.isfinite(reward) or not math.isfinite(task_reward):
            raise SystemExit(f"[fatal] non-finite reward at {path}:{line_no}")

        expected_task_reward = 1.0 if bool(row.get("passed", False)) else -1.0
        if abs(task_reward - expected_task_reward) > 1e-6:
            raise SystemExit(
                f"[fatal] per-turn task_reward mismatch at {path}:{line_no}: "
                f"passed={row.get('passed')!r}, task_reward={task_reward}"
            )
        if row.get("judge_failed") is True:
            failed += 1

        agents[str(row.get("agent_id", ""))] += 1
        groups.add((row.get("problem_id"), row.get("language"), row.get("rollout_idx")))

if rows == 0:
    raise SystemExit(f"[fatal] rollout is empty: {path}")
if failed and not allow_judge_failed:
    raise SystemExit(
        f"[fatal] rollout contains {failed} judge_failed rows; rerun rejudge or set "
        "ALLOW_JUDGE_FAILED=1 explicitly"
    )

print(
    f"[ok] rollout validation: rows={rows} groups={len(groups)} "
    f"judge_failed={failed} agents={dict(sorted(agents.items()))}"
)
PY

case " $AGENTS " in
    *" A1 "*|*" A2 "*|*" A3 "*) ;;
    *) echo "[fatal] AGENTS must contain at least one of A1, A2, A3"; exit 1 ;;
esac

for FRESH_AGENT in $FRESH_LORA_AGENTS; do
    case "$FRESH_AGENT" in
        A1|A2|A3) ;;
        *) echo "[fatal] unknown agent in FRESH_LORA_AGENTS: $FRESH_AGENT"; exit 1 ;;
    esac
done

for AGENT in $AGENTS; do
    case "$AGENT" in
        A1|A2|A3) ;;
        *) echo "[fatal] unknown agent in AGENTS: $AGENT"; exit 1 ;;
    esac
    MODEL_VAR="MODEL_$AGENT"
    SFT_VAR="SFT_$AGENT"
    MODEL_PATH="${!MODEL_VAR}"
    if [[ " $FRESH_LORA_AGENTS " == *" $AGENT "* ]]; then
        SFT_PATH=""
    else
        SFT_PATH="${!SFT_VAR}"
    fi
    [[ -d "$MODEL_PATH" ]] || { echo "[fatal] $AGENT model not found: $MODEL_PATH"; exit 1; }
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "[fatal] $AGENT model config missing: $MODEL_PATH/config.json"; exit 1; }
    if [[ -n "$SFT_PATH" ]]; then
        [[ -d "$SFT_PATH" ]] || { echo "[fatal] $AGENT SFT adapter not found: $SFT_PATH"; exit 1; }
        [[ -f "$SFT_PATH/adapter_model.safetensors" ]] || { echo "[fatal] $AGENT adapter weights missing: $SFT_PATH/adapter_model.safetensors"; exit 1; }
        [[ -f "$SFT_PATH/adapter_config.json" ]] || { echo "[fatal] $AGENT adapter config missing: $SFT_PATH/adapter_config.json"; exit 1; }
        "$PYTHON_BIN" - "$AGENT" "$MODEL_PATH" "$SFT_PATH/adapter_config.json" "$LORA_RANK" "$LORA_ALPHA" <<'PY'
import json
import os
import sys

agent, model_path, config_path, rank, alpha = sys.argv[1:]
cfg = json.load(open(config_path, encoding="utf-8"))
recorded_base = cfg.get("base_model_name_or_path")
if not recorded_base or os.path.realpath(recorded_base) != os.path.realpath(model_path):
    raise SystemExit(
        f"[fatal] {agent} adapter base mismatch: config={recorded_base!r}, model={model_path!r}"
    )
if int(cfg.get("r", -1)) != int(rank):
    raise SystemExit(
        f"[fatal] {agent} adapter rank mismatch: config r={cfg.get('r')}, LORA_RANK={rank}"
    )
if int(cfg.get("lora_alpha", -1)) != int(alpha):
    raise SystemExit(
        f"[fatal] {agent} adapter alpha mismatch: config lora_alpha={cfg.get('lora_alpha')}, LORA_ALPHA={alpha}"
    )
PY
    else
        echo "[ok] $AGENT will initialize a fresh LoRA on the base model"
    fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[ok] DRY_RUN=1: preflight complete; no training launched"
    exit 0
fi

# ============================================================
# Train each agent
# ============================================================
for AGENT in $AGENTS; do
    MODEL_VAR="MODEL_$AGENT"
    SFT_VAR="SFT_$AGENT"
    MODEL_PATH="${!MODEL_VAR}"
    if [[ " $FRESH_LORA_AGENTS " == *" $AGENT "* ]]; then
        SFT_PATH=""
    else
        SFT_PATH="${!SFT_VAR}"
    fi
    AGENT_OUT="$RL_RUNS_DIR/$AGENT"

    echo "================================================================"
    echo "[RL] Training $AGENT"
    echo "  base:        $MODEL_PATH"
    if [[ -n "$SFT_PATH" ]]; then
        echo "  sft adapter: $SFT_PATH"
    else
        echo "  sft adapter: <none> (fresh LoRA)"
    fi
    echo "  rollout:     $ROLLOUT"
    echo "  out:         $AGENT_OUT"
    echo "================================================================"
    mkdir -p "$AGENT_OUT"

    LAUNCH_ARGS=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
    if [[ "$NUM_GPUS" != "1" ]]; then
        LAUNCH_ARGS+=(--multi_gpu)
    fi

    TRAIN_ARGS=(
        --agent              "$AGENT"
        --rollout            "$ROLLOUT"
        --reward-field       "$REWARD_FIELD"
        --out-dir            "$AGENT_OUT"
        --model-name-or-path "$MODEL_PATH"
        --kl-coef            "$KL_COEF"
        --num-epochs         "$NUM_EPOCHS"
        --max-steps          "$MAX_STEPS"
        --learning-rate      "$LR"
        --seed               "$SEED"
        --per-device-batch-size "$PER_DEVICE_BATCH"
        --gradient-accumulation-steps "$GRAD_ACCUM"
        --max-seq-length     "$MAX_SEQ"
        --lora-rank          "$LORA_RANK"
        --lora-alpha         "$LORA_ALPHA"
        --lora-dropout       "$LORA_DROPOUT"
        --warmup-ratio       "$WARMUP_RATIO"
        --logging-steps      "$LOGGING_STEPS"
        --save-steps         "$SAVE_STEPS"
        --save-total-limit   "$SAVE_TOTAL_LIMIT"
        --report-to          "$REPORT_TO"
        --gradient-checkpointing
        --bf16
    )
    if [[ -n "$SFT_PATH" ]]; then
        TRAIN_ARGS+=(--sft-adapter "$SFT_PATH")
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
for AGENT in $AGENTS; do
    echo "Final adapter:  $RL_RUNS_DIR/$AGENT/final"
done
echo "TensorBoard:    tensorboard --logdir $RL_RUNS_DIR"
echo "================================================================"
