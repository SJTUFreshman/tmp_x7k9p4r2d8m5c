#!/usr/bin/env bash
set -euo pipefail

# Train one LoRA adapter per JCA role from build_conifer_sft.py output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${ROOT}/12_sft_data/default}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_sft}"
RUN_DIR="${RUN_DIR:-${ROOT}/11_runs/sft/${RUN_ID}}"
AGENTS="${AGENTS:-A1 A2 A3}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-2e-5}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-12288}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_POLICY="${SOURCE_POLICY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

[[ -d "${SFT_DATA_DIR}" ]] || { echo "[fatal] SFT_DATA_DIR not found: ${SFT_DATA_DIR}" >&2; exit 1; }
[[ "${MIXED_PRECISION}" == bf16 || "${MIXED_PRECISION}" == fp16 || "${MIXED_PRECISION}" == no ]] || { echo "[fatal] bad MIXED_PRECISION" >&2; exit 1; }
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/launch.log") 2>&1

cat >"${RUN_DIR}/config.env" <<EOF
RUN_ID=${RUN_ID}
SFT_DATA_DIR=${SFT_DATA_DIR}
RUN_DIR=${RUN_DIR}
MODEL_A1=${MODEL_A1}
MODEL_A2=${MODEL_A2}
MODEL_A3=${MODEL_A3}
SOURCE_POLICY=${SOURCE_POLICY}
NUM_GPUS=${NUM_GPUS}
NUM_EPOCHS=${NUM_EPOCHS}
LR=${LR}
MAX_SEQ=${MAX_SEQ}
EOF

echo "Conifer SFT: data=${SFT_DATA_DIR} run=${RUN_DIR} agents=${AGENTS}"
for agent in ${AGENTS}; do
  data_file="${SFT_DATA_DIR}/${agent}.jsonl"
  model_var="MODEL_${agent}"
  model_path="${!model_var}"
  out_dir="${RUN_DIR}/${agent}"
  [[ -s "${data_file}" ]] || { echo "[skip] ${agent}: missing ${data_file}"; continue; }
  [[ -d "${model_path}" ]] || { echo "[skip] ${agent}: missing model ${model_path}"; continue; }
  if [[ "${SKIP_COMPLETED}" == 1 && -s "${out_dir}/final/adapter_model.safetensors" ]]; then
    echo "[resume] ${agent}: final adapter already exists; skipping"
    continue
  fi
  cmd=("${ACCELERATE}" launch --num_processes "${NUM_GPUS}" --mixed_precision "${MIXED_PRECISION}")
  [[ "${NUM_GPUS}" == 1 ]] || cmd+=(--multi_gpu)
  cmd+=("${PROJECT_ROOT}/scripts/sft_train.py"
    --agent "${agent}" --data "${data_file}" --out-dir "${out_dir}"
    --model-name-or-path "${model_path}" --eval-fraction "${EVAL_FRACTION}"
    --split-group-by problem_id --num-epochs "${NUM_EPOCHS}" --learning-rate "${LR}"
    --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}"
    --per-device-train-batch-size "${PER_DEVICE_BATCH}" --per-device-eval-batch-size "${PER_DEVICE_BATCH}"
    --gradient-accumulation-steps "${GRAD_ACCUM}" --max-seq-length "${MAX_SEQ}"
    --report-to "${REPORT_TO}" --gradient-checkpointing --bf16 --no-load-best-model-at-end)
  printf '[command] '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "${DRY_RUN}" != 1 ]]; then
    PYTHONPATH="${PROJECT_ROOT}" "${cmd[@]}"
  fi
done
echo "[ok] Conifer SFT launcher complete"
