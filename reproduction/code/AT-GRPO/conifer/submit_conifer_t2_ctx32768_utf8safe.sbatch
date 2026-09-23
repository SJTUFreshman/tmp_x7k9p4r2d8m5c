#!/usr/bin/env bash
#SBATCH --job-name=atgrpo-conifer-t2-safe
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --time=0
#SBATCH --output=/data/run01/scwb515/yangrunde/atgrpo_conifer_seedsweep_20260922/logs/slurm-%j.log
set -Eeo pipefail
source /etc/profile
set -u
module load miniforge3/25.11.0-1
module load cuda/12.8
ROOT=/data/run01/scwb515/yangrunde/atgrpo_conifer_seedsweep_20260922
export PYTHON_BIN=/data/apps/miniforge3/25.11.0-1/bin/python
export PYTHONPATH="$ROOT:/data/home/scwb515/run/yangrunde/magrpo_baseline/resources/python:/data/home/scwb515/run/.local/lib/python3.12/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/acsw-${SLURM_JOB_ID}" VLLM_RPC_BASE_PATH="/tmp/acsw-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$ROOT/cache" TRITON_HOME=/data/run01/scwb515
export TRITON_CACHE_DIR="$ROOT/cache/triton" CUDA_CACHE_PATH="$ROOT/cache/nv"
export MPLCONFIGDIR="$ROOT/cache/matplotlib" XDG_CONFIG_HOME="$ROOT/config"
export VLLM_CONFIG_ROOT="$ROOT/config/vllm" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$VLLM_CONFIG_ROOT"
cd "$ROOT"
exec "$PYTHON_BIN" -u scripts/eval_conifer_seedsweep.py \
  --config configs/conifer_seedsweep_t2_ctx32768.yaml \
  --run-dir /data/run01/scwb515/yangrunde/atgrpo_baseline/runs/conifer/paracloud_v1 \
  --output "$ROOT/results_t2_ctx32768_utf8safe" --iteration 44 --first-seed 1 --max-concurrency 24
