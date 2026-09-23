#!/usr/bin/env bash
set -Eeo pipefail
source /etc/profile
set -u
module load miniforge3/25.11.0-1
module load cuda/12.8
module load apptainer/1.4.5
ROOT=/data/run01/scwb515/yangrunde/maporl_baseline
SHARED_RESOURCES=/data/run01/scwb515/yangrunde/magrpo_baseline/resources
export PYTHON_BIN=/data/apps/miniforge3/25.11.0-1/bin/python
export VLLM_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SHARED_RESOURCES/python:/data/home/scwb515/run/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export MAPORL_GSM_HARD_DATA_ROOT="$SHARED_RESOURCES/data/Math/data/GSM-HARD"
export MAPORL_MATH_DATA_ROOT="$SHARED_RESOURCES/data/Math/data/MATH"
export MAPORL_MUSIQUE_DATA_ROOT="$SHARED_RESOURCES/data/musique_data"
export MAPORL_CONIFER_DATA_ROOT="$SHARED_RESOURCES/data/conifer_training_hub/01_dataset/processed"
export MAPORL_MULTIPL_E_DATA_ROOT="$SHARED_RESOURCES/data/Code/multipl_e_8lang_benchmark/splits/unified_train70_test30_seed7658190907657085414"
export MULTIPLE_APPTAINER_IMAGE="$SHARED_RESOURCES/containers/multipl-e-evaluation.sif"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$ROOT/resources/cache"
export APPTAINER_CACHEDIR="$ROOT/resources/containers/cache"
export APPTAINER_TMPDIR="/tmp/maporl-apptainer-${SLURM_JOB_ID:-manual}"
export TMPDIR="$ROOT/resources/tmp/${SLURM_JOB_ID:-manual}"
# --- MAPORL_NO_HOME_CACHE: keep every cache off the 1 GiB $HOME quota ---
# Triton/CUDA/HF/Torch ignore XDG_CACHE_HOME and default to $HOME, which
# exhausted the home quota and killed vLLM EngineCore with OSError 122.
export TRITON_HOME=/data/run01/scwb515
export TRITON_CACHE_DIR=/data/run01/scwb515/.triton/cache
export CUDA_CACHE_PATH="$ROOT/resources/cache/nv"
export HF_HOME="$ROOT/resources/cache/huggingface"
export TORCH_HOME="$ROOT/resources/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/resources/cache/inductor"
export MPLCONFIGDIR="$ROOT/resources/cache/matplotlib"
export XDG_CONFIG_HOME="$ROOT/resources/config"
export VLLM_CONFIG_ROOT="$ROOT/resources/config/vllm"
# vLLM telemetry thread appended to ~/.config/vllm/usage_stats.json (1.3 MB
# and growing) and raised OSError 122 once home filled. Opt out entirely.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# --- end MAPORL_NO_HOME_CACHE ---
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CONFIG_HOME" "$VLLM_CONFIG_ROOT"
cd "$ROOT"

cleanup_maporl_job_tmp() {
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || return 1
    [[ "$(realpath -m -- "$TMPDIR")" == "$ROOT/resources/tmp/$SLURM_JOB_ID" ]] || return 1
    [[ "$(realpath -m -- "$APPTAINER_TMPDIR")" == "/tmp/maporl-apptainer-$SLURM_JOB_ID" ]] || return 1
    rm -rf -- "$TMPDIR" "$APPTAINER_TMPDIR"
}
