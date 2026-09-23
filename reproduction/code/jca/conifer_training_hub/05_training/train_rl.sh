#!/usr/bin/env bash
set -euo pipefail

# Signed reward-weighted regression (RWR) for Conifer turn records.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"
RL_DATA="${RL_DATA:-${ROOT}/13_rl_data/default.jsonl}"
SFT_ROOT="${SFT_ROOT:-${ROOT}/12_sft_data/default_sft_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_rl}"
RUN_DIR="${RUN_DIR:-${ROOT}/11_runs/rl/${RUN_ID}}"
AGENTS="${AGENTS:-A1 A2 A3}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
REWARD_FIELD="${REWARD_FIELD:-reward}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-5e-6}"
KL_COEF="${KL_COEF:-0.05}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_SEQ="${MAX_SEQ:-6144}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DRY_RUN="${DRY_RUN:-0}"
NO_SFT="${NO_SFT:-0}"
REQUIRE_SFT_INIT="${REQUIRE_SFT_INIT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
SAVE_STEPS="${SAVE_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:-0}"
RL_MAX_RETRIES="${RL_MAX_RETRIES:-2}"
RL_RETRY_DELAY="${RL_RETRY_DELAY:-20}"
JCA_RWR_LOGITS_CHUNK_SIZE="${JCA_RWR_LOGITS_CHUNK_SIZE:-32}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"
export PYTORCH_CUDA_ALLOC_CONF NCCL_CUMEM_ENABLE NCCL_P2P_DISABLE NCCL_IB_DISABLE \
  TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ENABLE_MONITORING \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC JCA_RWR_LOGITS_CHUNK_SIZE

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

[[ -s "${RL_DATA}" ]] || { echo "[fatal] RL_DATA not found: ${RL_DATA}" >&2; exit 1; }
case "${NO_SFT}" in 0|1) ;; *) echo "[fatal] NO_SFT must be 0 or 1" >&2; exit 1;; esac
case "${REQUIRE_SFT_INIT}" in 0|1) ;; *) echo "[fatal] REQUIRE_SFT_INIT must be 0 or 1" >&2; exit 1;; esac
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/launch.log") 2>&1
echo "Conifer RL: data=${RL_DATA} reward=${REWARD_FIELD} run=${RUN_DIR}"
echo "Conifer RL settings: batch=${PER_DEVICE_BATCH} accum=${GRAD_ACCUM} max_seq=${MAX_SEQ} save_steps=${SAVE_STEPS} resume=${RESUME_FROM_CHECKPOINT:-none} retries=${RL_MAX_RETRIES}"
echo "Conifer NCCL settings: cumem=${NCCL_CUMEM_ENABLE} p2p_disable=${NCCL_P2P_DISABLE} ib_disable=${NCCL_IB_DISABLE} async=${TORCH_NCCL_ASYNC_ERROR_HANDLING} monitoring=${TORCH_NCCL_ENABLE_MONITORING} heartbeat=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}s logits_chunk=${JCA_RWR_LOGITS_CHUNK_SIZE}"

for agent in ${AGENTS}; do
  model_var="MODEL_${agent}"; model_path="${!model_var}"
  out_dir="${RUN_DIR}/${agent}"
  sft_adapter="${SFT_ROOT}/${agent}/final"
  if [[ "${SKIP_COMPLETED}" == 1 && -s "${out_dir}/final/adapter_model.safetensors" && -s "${out_dir}/final/rl_train_summary.json" ]]; then
    echo "[resume] ${agent}: final adapter already exists; skipping"
    continue
  fi
  [[ -d "${model_path}" ]] || { echo "[skip] ${agent}: missing model ${model_path}"; continue; }
  cmd=("${ACCELERATE}" launch --num_processes "${NUM_GPUS}" --mixed_precision "${MIXED_PRECISION}")
  [[ "${NUM_GPUS}" == 1 ]] || cmd+=(--multi_gpu)
  cmd+=("${PROJECT_ROOT}/scripts/rl_train.py" --agent "${agent}" --rollout "${RL_DATA}"
    --reward-field "${REWARD_FIELD}" --out-dir "${out_dir}" --model-name-or-path "${model_path}"
    --kl-coef "${KL_COEF}" --num-epochs "${NUM_EPOCHS}" --learning-rate "${LR}"
    --per-device-batch-size "${PER_DEVICE_BATCH}" --gradient-accumulation-steps "${GRAD_ACCUM}"
    --max-seq-length "${MAX_SEQ}" --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}" --report-to "${REPORT_TO}" --gradient-checkpointing --bf16)
  if [[ "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    cmd+=(--save-steps "${SAVE_STEPS}")
  fi
  if [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    cmd+=(--max-steps "${MAX_STEPS}")
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    cmd+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
  if [[ "${NO_SFT}" != 1 ]]; then
    if [[ "${REQUIRE_SFT_INIT}" == 1 ]]; then
      [[ -s "${sft_adapter}/adapter_model.safetensors" ]] || {
        echo "[fatal] ${agent}: RL must initialize from SFT adapter: ${sft_adapter}" >&2
        exit 1
      }
    else
      [[ -d "${sft_adapter}" ]] || { echo "[skip] ${agent}: missing SFT adapter ${sft_adapter}"; continue; }
    fi
    cmd+=(--sft-adapter "${sft_adapter}")
  fi
  printf '[command] '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "${DRY_RUN}" != 1 ]]; then
    attempt=0
    while true; do
      if PYTHONPATH="${PROJECT_ROOT}" "${cmd[@]}"; then
        break
      else
        status=$?
      fi
      if (( attempt >= RL_MAX_RETRIES )); then
        echo "[fatal] ${agent}: RL failed after $((attempt + 1)) attempt(s)" >&2
        exit "${status}"
      fi
      attempt=$((attempt + 1))
      echo "[retry] ${agent}: RL failed (exit=${status}); retry ${attempt}/${RL_MAX_RETRIES} after ${RL_RETRY_DELAY}s"
      sleep "${RL_RETRY_DELAY}"
    done
  fi
done
echo "[ok] Conifer RL launcher complete"
