#!/usr/bin/env bash
set -Eeuo pipefail

# Single entrypoint: DATASET=gsm_hard bash scripts/run_atgrpo.sh
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
PRESCREEN_FILE="${PRESCREEN_FILE:-${ROOT}/runs/_shared/prescreen/${DATASET}.jsonl}"
SKIP_GPU_GATE="${SKIP_GPU_GATE:-0}"
GPU_IDLE_MAX_MEMORY_MIB="${GPU_IDLE_MAX_MEMORY_MIB:-2000}"
GPU_GATE_TIMEOUT_S="${GPU_GATE_TIMEOUT_S:-0}"   # 0 = wait forever
EVAL_AFTER="${EVAL_AFTER:-1}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-16}"
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
from atgrpo.config import load_config
c = load_config('${CONFIG}')
print(f'  task={c.task} K={c.atgrpo.group_size_K} branch={c.atgrpo.branch_mode} B={c.atgrpo.prompts_per_iter} '
      f'gpu_plan={c.serving.gpu_plan} '
      f'lora_naming={c.serving.lora_naming} prefix_cache={c.serving.enable_prefix_caching}')
" || fail "config validation failed"

CONFIG_TASK=$("${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${ROOT}')
from atgrpo.config import load_config
print(load_config('${CONFIG}').task)
") || fail "could not read config task"
[[ "${CONFIG_TASK}" == "${DATASET}" ]] || fail \
  "DATASET=${DATASET} does not match config task=${CONFIG_TASK}"

PRESCREEN_ENABLED=$("${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${ROOT}')
from atgrpo.config import load_config
print('1' if load_config('${CONFIG}').pool.prescreen else '0')
") || fail "could not read prescreen setting"
if [[ "${PRESCREEN_ENABLED}" == 1 ]]; then
  [[ -f "${PRESCREEN_FILE}" && -s "${PRESCREEN_FILE}" ]] || fail \
    "prescreen is enabled but allowlist is missing or empty: ${PRESCREEN_FILE}"
  if ! ALLOW_COUNT=$(ATGRPO_ROOT="${ROOT}" ATGRPO_CONFIG="${CONFIG}" \
    PRESCREEN_PATH="${PRESCREEN_FILE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ATGRPO_ROOT", "."))
from atgrpo.config import load_config
from atgrpo.pool import expected_prescreen_count, load_allowlist
from atgrpo.tasks import build_task

config = load_config(os.environ["ATGRPO_CONFIG"])
task = build_task(
    config.task,
    **({"reward_mode": config.reward_mode} if config.task == "multipl_e" else {}),
)
problems = task.load("train")
allow = load_allowlist(
    Path(os.environ["PRESCREEN_PATH"]),
    expected_group_size=config.atgrpo.group_size_K,
    expected_count=expected_prescreen_count(
        config.pool.prescreen_sample, len(problems)
    ),
    expected_keep_band=config.pool.keep_band,
    source_ids={problem.problem_id for problem in problems},
)
if not allow:
    raise SystemExit("allowlist contains no kept prompts")
print(len(allow))
PY
  ); then
    fail "invalid prescreen allowlist: ${PRESCREEN_FILE}"
  fi
  log "using prescreen allowlist: ${PRESCREEN_FILE} (${ALLOW_COUNT} prompts)"
fi

# --- 4. models -------------------------------------------------------------
log "checking model directories"
"${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${ROOT}')
from pathlib import Path
from atgrpo.config import load_config
c = load_config('${CONFIG}')
missing = [p for p in c.serving.models.values() if not Path(p).is_dir()]
if missing:
    print(f'[fatal] missing model dirs: {missing}'); sys.exit(1)
" || exit 1

# --- 5. MultiPL-E only: execution container -------------------------------
if [[ "${DATASET}" == "multipl_e" ]]; then
  IMAGE="${MULTIPLE_IMAGE:-multipl-e-evaluation:jca-current}"
  if [[ -n "${MULTIPLE_APPTAINER_IMAGE:-}" ]]; then
    [[ -f "${MULTIPLE_APPTAINER_IMAGE}" ]] || fail "Apptainer image not found: ${MULTIPLE_APPTAINER_IMAGE}"
    log "checking MultiPL-E Apptainer image: ${MULTIPLE_APPTAINER_IMAGE}"
  else
    log "checking MultiPL-E docker image"
    command -v docker >/dev/null 2>&1 || fail "docker unavailable; set MULTIPLE_APPTAINER_IMAGE to a local .sif"
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || fail "docker image ${IMAGE} not available; build it or set MULTIPLE_APPTAINER_IMAGE"
  fi
fi

# --- 6. GPU gate -----------------------------------------------------------
if [[ "${SKIP_GPU_GATE}" != 1 ]]; then
  log "waiting for GPUs to go idle (threshold ${GPU_IDLE_MAX_MEMORY_MIB} MiB)"
  START_TS=$(date +%s)
  while true; do
    GPU_STATUS=""
    if ! GPU_STATUS=$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits 2>/dev/null); then
      fail "nvidia-smi query failed; refusing to assume GPUs are idle"
    fi
    [[ -n "${GPU_STATUS}" ]] || fail "nvidia-smi returned no GPU status"
    BUSY=$(awk -F', ' -v t="${GPU_IDLE_MAX_MEMORY_MIB}" \
      '$2 > t {printf "%s ", $1}' <<<"${GPU_STATUS}")
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
[[ "${MAX_ITERATIONS}" != 0 ]] && ARGS+=(--max-iterations "${MAX_ITERATIONS}")
if [[ "${MAX_MODEL_LEN}" != 0 ]]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
[[ -f "${PRESCREEN_FILE}" ]] && ARGS+=(--prescreen "${PRESCREEN_FILE}")

log "training: ${DATASET}"
"${PYTHON_BIN}" "${SELF_DIR}/train_atgrpo.py" "${ARGS[@]}"

# --- 8. evaluate -----------------------------------------------------------
if [[ "${EVAL_AFTER}" == 1 ]]; then
  log "evaluating: ${DATASET}"
  EVAL_ARGS=(
    --config "${CONFIG}"
    --iteration last
    --max-concurrency "${EVAL_MAX_CONCURRENCY:-16}"
    ${MAX_MODEL_LEN:+--max-model-len ${MAX_MODEL_LEN}}
  )
  [[ -n "${RUN_DIR}" ]] && EVAL_ARGS+=(--run-dir "${RUN_DIR}")
  "${PYTHON_BIN}" "${SELF_DIR}/eval_atgrpo.py" "${EVAL_ARGS[@]}"
fi

log "done: ${DATASET}"
