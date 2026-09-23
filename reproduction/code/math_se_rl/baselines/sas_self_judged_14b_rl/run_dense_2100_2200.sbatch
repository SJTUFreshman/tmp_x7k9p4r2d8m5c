#!/bin/bash
#SBATCH --partition=hp_a800
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00
#SBATCH --job-name=math-sas14b-dense-2100-2200
#SBATCH --output=/data/home/scwb515/run/yangrunde/math_specific_sft_rl_v1/slurm/logs/math-sas14b-dense-2100-2200-%j.out
#SBATCH --error=/data/home/scwb515/run/yangrunde/math_specific_sft_rl_v1/slurm/logs/math-sas14b-dense-2100-2200-%j.err

set -Eeuo pipefail

EXPERIMENT_ROOT="/data/home/scwb515/run/yangrunde/math_specific_sft_rl_v1"
RUNTIME_ENV_ROOT="/data/home/scwb515/.conda/envs/swift"
export HOME=/data/home/scwb515 USER=scwb515 SLURM_EXPORT_ENV=ALL
source /etc/profile.d/modules.sh
module load miniforge3/25.11.0-1
module load cuda/12.8
export CONDA_PREFIX="$RUNTIME_ENV_ROOT" CONDA_DEFAULT_ENV="$RUNTIME_ENV_ROOT"
export CUDA_HOME="${CUDA_HOME:-/data/apps/cuda/12.8}" CUDA_PATH="${CUDA_PATH:-/data/apps/cuda/12.8}"
export PATH="$CUDA_HOME/bin:$RUNTIME_ENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset PYTHONPATH PYTHONHOME
export RUNTIME_ENV_ROOT OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0
export https_proxy="${https_proxy:-}"
export http_proxy="${http_proxy:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}" HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export NCCL_RAS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true VLLM_USE_V1=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

task_tmpdir="$(mktemp -d "/tmp/${USER}.d2100.${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$task_tmpdir" && "$task_tmpdir" == "/tmp/${USER}.d2100.${SLURM_JOB_ID}."* ]]; then
    rm -rf -- "$task_tmpdir"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM
export TMPDIR="$task_tmpdir" RAY_TMPDIR="$task_tmpdir/ray" TORCHINDUCTOR_CACHE_DIR="$task_tmpdir/torchinductor"
mkdir -p "$RAY_TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"
ulimit -n 65536

full_gpu_ids="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"
if [[ -z "$full_gpu_ids" ]]; then
  count="${SLURM_GPUS_ON_NODE:-8}"
  full_gpu_ids="$(seq -s, 0 $((count - 1)))"
fi
full_gpu_ids="${full_gpu_ids//[[:space:]]/}"
gpu_count="$(awk -F',' 'NF {print NF; exit}' <<<"$full_gpu_ids")"
[[ "$gpu_count" == "8" ]] || {
  echo "[fatal] expected an eight-GPU allocation, got $gpu_count: $full_gpu_ids" >&2
  exit 1
}
train_gpu_ids="$(cut -d, -f1-2 <<<"$full_gpu_ids")"
export TRAIN_GPU_IDS="$train_gpu_ids" TRAIN_NUM_GPUS=2
export EVAL_GPU_IDS="$full_gpu_ids" EVAL_NUM_GPUS=8
export GPU_GUARD_USE_ALLOCATION=1 WAIT_FOR_IDLE_GPUS=0

echo "[slurm] job=$SLURM_JOB_ID node=$(hostname) train_gpus=$TRAIN_GPU_IDS eval_gpus=$EVAL_GPU_IDS"
cd "$EXPERIMENT_ROOT"
bash "$EXPERIMENT_ROOT/baselines/sas_self_judged_14b_rl/run_dense_2100_2200.sh"
