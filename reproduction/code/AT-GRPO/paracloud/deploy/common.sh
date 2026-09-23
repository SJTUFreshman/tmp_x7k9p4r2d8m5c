#!/usr/bin/env bash
# Shared ParaCloud Zhongwei-1 environment for the AT-GRPO baseline.
# Mirrors magrpo_baseline/deploy/common.sh; see that file for the history behind
# the cache redirections.
set -Eeo pipefail
source /etc/profile
set -u

module load miniforge3/25.11.0-1
module load cuda/12.8
module load apptainer/1.4.5

ROOT=/data/home/scwb515/run/yangrunde/atgrpo_baseline
export ROOT
export PYTHON_BIN=/data/apps/miniforge3/25.11.0-1/bin/python
export VLLM_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

# $ROOT makes both `atgrpo` and the flat `vendor` package importable.
export PYTHONPATH="$ROOT:/data/home/scwb515/run/yangrunde/magrpo_baseline/resources/python:/data/home/scwb515/run/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Compute nodes have no network: every model and dataset must resolve locally.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MultiPL-E execution reward runs through Apptainer; there is no Docker here.
export MULTIPLE_APPTAINER_IMAGE="$ROOT/resources/containers/multipl-e-evaluation.sif"
export APPTAINER_CACHEDIR="$ROOT/resources/containers/cache"
export APPTAINER_TMPDIR="/tmp/atgrpo-apptainer-${SLURM_JOB_ID:-manual}"
export TMPDIR="$ROOT/resources/tmp/${SLURM_JOB_ID:-manual}"

# --- keep caches off the 1 GiB $HOME quota -------------------------------
# Triton ignores XDG_CACHE_HOME and defaults to $HOME; a full home quota kills
# vLLM EngineCore with OSError 122 from the LoRA-kernel compile cache.
export XDG_CACHE_HOME="$ROOT/resources/cache"
export TRITON_HOME=/data/run01/scwb515
export TRITON_CACHE_DIR=/data/run01/scwb515/.triton/cache
export CUDA_CACHE_PATH="$ROOT/resources/cache/nv"
export MPLCONFIGDIR="$ROOT/resources/cache/matplotlib"
export XDG_CONFIG_HOME="$ROOT/resources/config"
export VLLM_CONFIG_ROOT="$ROOT/resources/config/vllm"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# --- end cache redirection -----------------------------------------------

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$XDG_CONFIG_HOME" \
  "$VLLM_CONFIG_ROOT" "$ROOT/slurm/logs" "$ROOT/slurm/configs"
cd "$ROOT"
