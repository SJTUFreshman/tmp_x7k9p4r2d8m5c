#!/usr/bin/env bash
# GSM correction-balanced protocol SFT launcher (independent copy).
#
# Uses the existing JCA SFT stack to train LoRA adapters for A1/A2/A3 from
# prebuilt GSM protocol data. By default it trains all three agents
# sequentially, each using all 8 GPUs through accelerate DDP.
#
# DATA_DIR must point to newly rebuilt, validated GSM protocol data. This
# launcher intentionally has no default so the rejected 0720 dataset cannot be
# selected accidentally.
#
# Outputs:
#   /data/wangyuheng/jca/sft_runs/<TAG>/{A1,A2,A3}/final
#
# Example:
#   DATA_DIR=/data/wangyuheng/jca/sft_data/<clean_tag> \
#   bash /data/wangyuheng/jca/scripts/run_gsm_protocol_sft.sh
#
#   TAG=gsm_proto_warmup_0720_v2 \
#   AGENTS="A1 A2 A3" \
#   NUM_EPOCHS=3 \
#   LR=1e-5 \
#   bash /data/wangyuheng/jca/scripts/run_gsm_protocol_sft.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
SFT_TRAIN_SH="${SFT_TRAIN_SH:-/data/wangyuheng/jca/scripts/sft_train.sh}"
PY="${PY:-/data/conda_envs/qwen35/bin/python}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"
TRAIN_PROJ_ROOT="${TRAIN_PROJ_ROOT:-/data/wangyuheng}"

DATA_DIR="${DATA_DIR:-}"
TAG="${TAG:-gsm_corr30_balanced_v1}"

# Train all three by default. Override to a subset if needed.
AGENTS="${AGENTS:-A1 A2 A3}"

# Conservative warmup defaults for GSM protocol data.
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-1e-5}"
EVAL_FRACTION="${EVAL_FRACTION:-0.05}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-4096}"
MAX_TOKENIZATION_DROP_RATIO="${MAX_TOKENIZATION_DROP_RATIO:-0.05}"
MIN_SAMPLES_PER_AGENT="${MIN_SAMPLES_PER_AGENT:-2200}"
ALLOW_FAILED_QUALITY_GATE="${ALLOW_FAILED_QUALITY_GATE:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
REPORT_TO="${REPORT_TO:-tensorboard}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_GEN_SAMPLES="${NUM_GEN_SAMPLES:-4}"
PER_AGENT_DATA="${PER_AGENT_DATA:-1}"
SFT_SPLIT_GROUP_BY="${SFT_SPLIT_GROUP_BY:-problem_id}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

# Optional init LoRA paths if you want to continue from an existing adapter.
INIT_LORA_A1="${INIT_LORA_A1:-}"
INIT_LORA_A2="${INIT_LORA_A2:-}"
INIT_LORA_A3="${INIT_LORA_A3:-}"

if [[ "$PER_AGENT_DATA" != "1" ]]; then
  echo "[fatal] GSM protocol SFT requires PER_AGENT_DATA=1"
  exit 1
fi
if [[ ! "$MIN_SAMPLES_PER_AGENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[fatal] MIN_SAMPLES_PER_AGENT must be positive"
  exit 1
fi
if [[ "$ALLOW_FAILED_QUALITY_GATE" != "0" && "$ALLOW_FAILED_QUALITY_GATE" != "1" ]]; then
  echo "[fatal] ALLOW_FAILED_QUALITY_GATE must be 0 or 1"
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "[fatal] Python executable not found: $PY"
  exit 1
fi
if [[ ! -x "$ACCELERATE" ]]; then
  echo "[fatal] accelerate executable not found: $ACCELERATE"
  exit 1
fi

if [[ -z "$DATA_DIR" ]]; then
  echo "[fatal] DATA_DIR is required; point it at newly rebuilt clean GSM protocol data"
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[fatal] DATA_DIR not found: $DATA_DIR"
  exit 1
fi

for req in all_agents.jsonl A1.jsonl A2.jsonl A3.jsonl stats.json; do
  if [[ ! -f "$DATA_DIR/$req" ]]; then
    echo "[fatal] missing required file: $DATA_DIR/$req"
    exit 1
  fi
done

for agent in $AGENTS; do
  case "$agent" in
    A1|A2|A3) ;;
    *)
      echo "[fatal] unknown agent in AGENTS: $agent"
      exit 1
      ;;
  esac
  data_file="$DATA_DIR/$agent.jsonl"
  model_var="MODEL_$agent"
  model_path="${!model_var}"
  if [[ ! -s "$data_file" ]]; then
    echo "[fatal] empty or missing identity-specific data: $data_file"
    exit 1
  fi
  sample_count=$(wc -l < "$data_file" | tr -d ' ')
  if [[ "$sample_count" -lt "$MIN_SAMPLES_PER_AGENT" ]]; then
    echo "[fatal] too few examples for $agent: $sample_count (need >=$MIN_SAMPLES_PER_AGENT)"
    exit 1
  fi
  if [[ ! -e "$model_path" ]]; then
    echo "[fatal] model path not found for $agent: $model_path"
    exit 1
  fi
done

"$PY" - "$DATA_DIR/stats.json" "$ALLOW_FAILED_QUALITY_GATE" <<'PY'
import json
import sys

stats_path = sys.argv[1]
allow_failed = sys.argv[2] == "1"
with open(stats_path, encoding="utf-8") as handle:
    stats = json.load(handle)
quality_gate = stats.get("quality_gate")
if not isinstance(quality_gate, dict):
    raise SystemExit(
        f"[fatal] {stats_path} has no quality_gate; rebuild data with the current GSM builder"
    )
if not quality_gate.get("passed", False):
    failed = [
        name
        for name, check in quality_gate.get("checks", {}).items()
        if not check.get("passed", False)
    ]
    message = "GSM SFT data failed quality gates: " + ", ".join(failed)
    if not allow_failed:
        raise SystemExit("[fatal] " + message)
    print("[preflight] WARNING: " + message)
    print("[preflight] continuing because ALLOW_FAILED_QUALITY_GATE=1")
else:
    print("[preflight] GSM SFT data quality gate: PASS")
PY

if grep -q '"trajectory_mapping_counts"' "$DATA_DIR/stats.json"; then
  echo "[fatal] DATA_DIR contains remapped agent identities; original IDs are required"
  exit 1
fi

if [[ ! -f "$SFT_TRAIN_SH" ]]; then
  echo "[fatal] sft_train.sh not found: $SFT_TRAIN_SH"
  exit 1
fi

echo "================================================================"
echo "GSM correction-balanced protocol SFT launcher"
echo "================================================================"
echo "time          = $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "host          = $(hostname)"
echo "project_root  = $PROJECT_ROOT"
echo "data_dir      = $DATA_DIR"
echo "tag           = $TAG"
echo "agents        = $AGENTS"
echo "model_a1      = $MODEL_A1"
echo "model_a2      = $MODEL_A2"
echo "model_a3      = $MODEL_A3"
echo "num_epochs    = $NUM_EPOCHS"
echo "lr            = $LR"
echo "eval_fraction = $EVAL_FRACTION"
echo "per_dev_batch = $PER_DEVICE_BATCH"
echo "grad_accum    = $GRAD_ACCUM"
echo "max_seq       = $MAX_SEQ"
echo "max_tok_drop  = $MAX_TOKENIZATION_DROP_RATIO"
echo "min_samples   = $MIN_SAMPLES_PER_AGENT per agent"
echo "allow_bad_gate= $ALLOW_FAILED_QUALITY_GATE"
echo "lora_rank     = $LORA_RANK"
echo "lora_alpha    = $LORA_ALPHA"
echo "lora_dropout  = $LORA_DROPOUT"
echo "num_gpus      = $NUM_GPUS"
echo "mixed_prec    = $MIXED_PRECISION"
echo "per_agent_data= $PER_AGENT_DATA"
echo "split_group_by= $SFT_SPLIT_GROUP_BY"
echo "python        = $PY"
echo "accelerate    = $ACCELERATE"
echo "================================================================"

cd "$PROJECT_ROOT"

PREBUILT_SFT_DATA_DIR="$DATA_DIR" \
PROJ_ROOT="$TRAIN_PROJ_ROOT" \
PY="$PY" \
ACCELERATE="$ACCELERATE" \
AGENTS="$AGENTS" \
NUM_EPOCHS="$NUM_EPOCHS" \
LR="$LR" \
EVAL_FRACTION="$EVAL_FRACTION" \
PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
GRAD_ACCUM="$GRAD_ACCUM" \
MAX_SEQ="$MAX_SEQ" \
MAX_TOKENIZATION_DROP_RATIO="$MAX_TOKENIZATION_DROP_RATIO" \
LOGGING_STEPS="$LOGGING_STEPS" \
EVAL_STEPS="$EVAL_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
LORA_RANK="$LORA_RANK" \
LORA_ALPHA="$LORA_ALPHA" \
LORA_DROPOUT="$LORA_DROPOUT" \
NUM_GPUS="$NUM_GPUS" \
MIXED_PRECISION="$MIXED_PRECISION" \
REPORT_TO="$REPORT_TO" \
LOAD_BEST_MODEL_AT_END="$LOAD_BEST_MODEL_AT_END" \
DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
NUM_GEN_SAMPLES="$NUM_GEN_SAMPLES" \
PER_AGENT_DATA="$PER_AGENT_DATA" \
SFT_SPLIT_GROUP_BY="$SFT_SPLIT_GROUP_BY" \
MODEL_A1="$MODEL_A1" \
MODEL_A2="$MODEL_A2" \
MODEL_A3="$MODEL_A3" \
INIT_LORA_A1="$INIT_LORA_A1" \
INIT_LORA_A2="$INIT_LORA_A2" \
INIT_LORA_A3="$INIT_LORA_A3" \
bash "$SFT_TRAIN_SH" "$TAG"

for agent in $AGENTS; do
  final_dir="$PROJECT_ROOT/sft_runs/$TAG/$agent/final"
  if [[ ! -s "$final_dir/adapter_config.json" ]]; then
    echo "[fatal] training did not produce a final adapter for $agent: $final_dir"
    exit 1
  fi
done
