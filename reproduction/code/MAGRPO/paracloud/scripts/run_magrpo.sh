#!/usr/bin/env bash
set -Eeuo pipefail

# Single entrypoint: DATASET=gsm_hard bash scripts/run_magrpo.sh
#
# Preflight gates run first and are all fatal. The GPU gate in particular will
# block until the requested devices are free -- that is intended, not a hang.

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"

DATASET="${DATASET:-gsm_hard}"
CONFIG="${CONFIG:-${ROOT}/configs/${DATASET}.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data/conda_envs/drb_py311_clean/bin/python}"
RUN_ID="${RUN_ID:-}"
RUN_DIR="${RUN_DIR:-}"
RESUME="${RESUME:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
IGNORE_WALL_CLOCK_LIMIT="${IGNORE_WALL_CLOCK_LIMIT:-0}"
PRESCREEN_FILE="${PRESCREEN_FILE:-${ROOT}/runs/_shared/prescreen/${DATASET}.jsonl}"
SKIP_GPU_GATE="${SKIP_GPU_GATE:-0}"
GPU_IDLE_MAX_MEMORY_MIB="${GPU_IDLE_MAX_MEMORY_MIB:-2000}"
GPU_GATE_TIMEOUT_S="${GPU_GATE_TIMEOUT_S:-0}"   # 0 = wait forever
EVAL_AFTER="${EVAL_AFTER:-1}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[fatal] %s\n' "$*" >&2; exit 1; }

[[ -f "${CONFIG}" ]] || fail "config not found: ${CONFIG}"

# --- 1. vendored-file drift ------------------------------------------------
log "checking vendored files"
"${PYTHON_BIN}" "${SELF_DIR}/vendor_sync.py" --check || \
  log "WARNING: vendored files differ from upstream (see message above)"

# --- 2. interpreters and packages -----------------------------------------
log "checking interpreters"
[[ -x "${PYTHON_BIN}" ]] || fail "python not executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY' || exit 1
import sys
missing = []
for module in ("torch", "transformers", "peft", "yaml"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print(f"[fatal] missing packages in training env: {missing}", file=sys.stderr)
    raise SystemExit(1)
PY
[[ -x "${VLLM_PYTHON}" ]] || fail "vLLM python not executable: ${VLLM_PYTHON}"

# --- 3. config validation (also enforces the prefix-cache guard) -----------
log "validating config"
"${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${ROOT}')
from magrpo.config import load_config
c = load_config('${CONFIG}')
print(f'  task={c.task} G={c.magrpo.group_size_G} B={c.magrpo.prompts_per_iter} '
      f'joint_mode={c.rollout.joint_mode} gpu_plan={c.serving.gpu_plan} '
      f'lora_naming={c.serving.lora_naming} prefix_cache={c.serving.enable_prefix_caching}')
" || fail "config validation failed"

# --- 4. models -------------------------------------------------------------
log "checking model directories"
"${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${ROOT}')
from pathlib import Path
from magrpo.config import load_config
c = load_config('${CONFIG}')
missing = [p for p in c.serving.models.values() if not Path(p).is_dir()]
if missing:
    print(f'[fatal] missing model dirs: {missing}'); sys.exit(1)
" || exit 1

# --- 5. MultiPL-E only: docker image --------------------------------------
if [[ "${DATASET}" == "multipl_e" ]]; then
  IMAGE="${MULTIPLE_IMAGE:-multipl-e-evaluation:jca-current}"
  if [[ -n "${MULTIPLE_APPTAINER_IMAGE:-}" ]]; then
    [[ -f "${MULTIPLE_APPTAINER_IMAGE}" ]] || fail "Apptainer image not found: ${MULTIPLE_APPTAINER_IMAGE}"
    log "checking MultiPL-E Apptainer image: ${MULTIPLE_APPTAINER_IMAGE}"
  else
    log "checking MultiPL-E docker image"
    command -v docker >/dev/null 2>&1 || fail "docker unavailable; set MULTIPLE_APPTAINER_IMAGE to a local .sif"
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || \
      fail "docker image ${IMAGE} not available; build it or set MULTIPLE_APPTAINER_IMAGE"
  fi
fi

# --- 6. GPU gate -----------------------------------------------------------
if [[ "${SKIP_GPU_GATE}" != 1 ]]; then
  log "waiting for GPUs to go idle (threshold ${GPU_IDLE_MAX_MEMORY_MIB} MiB)"
  START_TS=$(date +%s)
  while true; do
    BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v t="${GPU_IDLE_MAX_MEMORY_MIB}" '$2 > t {printf "%s ", $1}')
    [[ -z "${BUSY}" ]] && break
    if (( GPU_GATE_TIMEOUT_S > 0 )) && (( $(date +%s) - START_TS > GPU_GATE_TIMEOUT_S )); then
      fail "GPUs still busy after ${GPU_GATE_TIMEOUT_S}s: ${BUSY}"
    fi
    log "  GPUs busy: ${BUSY}(waiting)"
    sleep 60
  done
  log "GPUs idle"
fi

[[ "${DRY_RUN}" == 1 ]] && { log "dry run: preflight passed, stopping"; exit 0; }

# --- 7. train --------------------------------------------------------------
ARGS=(--config "${CONFIG}")
[[ -n "${RUN_DIR}" ]] && ARGS+=(--run-dir "${RUN_DIR}")
[[ "${RESUME}" == 1 ]] && ARGS+=(--resume)
[[ -n "${WARM_START_FROM:-}" ]] && ARGS+=(--warm-start-from "${WARM_START_FROM}")
[[ "${MAX_ITERATIONS}" != 0 ]] && ARGS+=(--max-iterations "${MAX_ITERATIONS}")
[[ "${IGNORE_WALL_CLOCK_LIMIT}" == 1 ]] && ARGS+=(--ignore-wall-clock-limit)
[[ -f "${PRESCREEN_FILE}" ]] && ARGS+=(--prescreen "${PRESCREEN_FILE}")

log "training: ${DATASET}"
"${PYTHON_BIN}" "${SELF_DIR}/train_magrpo.py" "${ARGS[@]}"

# --- 8. evaluate -----------------------------------------------------------
if [[ "${EVAL_AFTER}" == 1 ]]; then
  log "evaluating: ${DATASET}"
  EVAL_ARGS=(--config "${CONFIG}" --iteration last --include-base)
  EVAL_ARGS+=(--max-concurrency "${EVAL_MAX_CONCURRENCY:-16}")
  [[ -n "${RUN_DIR}" ]] && EVAL_ARGS+=(--run-dir "${RUN_DIR}")
  "${PYTHON_BIN}" "${SELF_DIR}/eval_magrpo.py" "${EVAL_ARGS[@]}"
fi

log "done: ${DATASET}"
