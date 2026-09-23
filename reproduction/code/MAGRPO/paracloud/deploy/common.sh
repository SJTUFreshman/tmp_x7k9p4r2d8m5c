#!/usr/bin/env bash
set -Eeo pipefail
source /etc/profile
set -u
module load miniforge3/25.11.0-1
module load cuda/12.8
module load apptainer/1.4.5
ROOT=/data/home/scwb515/run/yangrunde/magrpo_baseline
export PYTHON_BIN=/data/apps/miniforge3/25.11.0-1/bin/python
export VLLM_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT/resources/python:$ROOT:/data/home/scwb515/run/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MAGRPO_GSM_HARD_DATA_ROOT="$ROOT/resources/data/Math/data/GSM-HARD"
export MAGRPO_MATH_DATA_ROOT="$ROOT/resources/data/Math/data/MATH"
export MAGRPO_MUSIQUE_DATA_ROOT="$ROOT/resources/data/musique_data"
export MAGRPO_CONIFER_DATA_ROOT="$ROOT/resources/data/conifer_training_hub/01_dataset/processed"
export MAGRPO_MULTIPL_E_DATA_ROOT="$ROOT/resources/data/Code/multipl_e_8lang_benchmark/splits/unified_train70_test30_seed7658190907657085414"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$ROOT/resources/cache"
export APPTAINER_CACHEDIR="$ROOT/resources/containers/cache"
export APPTAINER_TMPDIR="/tmp/magrpo-apptainer-${SLURM_JOB_ID:-manual}"
export TMPDIR="$ROOT/resources/tmp/${SLURM_JOB_ID:-manual}"
# --- MAGRPO_NO_HOME_CACHE: keep caches off the 1 GiB $HOME quota ---
# Only .triton (26 MB, unbounded), .nv and .config actually sit on the home
# quota; .cache/.local are already symlinks to the project disk, so HF_HOME
# and TORCH_HOME are deliberately left alone (HF_HUB_OFFLINE=1 needs the
# existing datasets/modules cache there). Triton ignores XDG_CACHE_HOME and
# defaults to $HOME; a full home quota kills vLLM EngineCore with OSError 122.
export TRITON_HOME=/data/run01/scwb515
export TRITON_CACHE_DIR=/data/run01/scwb515/.triton/cache
export CUDA_CACHE_PATH="$ROOT/resources/cache/nv"
export MPLCONFIGDIR="$ROOT/resources/cache/matplotlib"
export XDG_CONFIG_HOME="$ROOT/resources/config"
export VLLM_CONFIG_ROOT="$ROOT/resources/config/vllm"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# --- end MAGRPO_NO_HOME_CACHE ---
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$XDG_CONFIG_HOME" "$VLLM_CONFIG_ROOT"
cd "$ROOT"
