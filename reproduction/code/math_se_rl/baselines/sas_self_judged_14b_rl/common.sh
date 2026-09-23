#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=config.env
export SCRIPT_DIR
set -a
source "${SCRIPT_DIR}/config.env"
set +a
source "${PROJECT_ROOT}/scripts/math_gpu_guard.sh"

export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

fatal() { echo "[fatal] $*" >&2; exit 1; }

_csv_count() {
  local value="${1//[[:space:]]/}"
  [[ -n "$value" ]] || return 1
  awk -F',' 'NF { count=0; for (i=1; i<=NF; i++) if ($i != "") count++; print count; exit }' <<<"$value"
}

resolve_runtime_gpu_config() {
  local visible="${GPU_IDS:-}"
  local allocated="${CUDA_VISIBLE_DEVICES:-}"
  [[ -n "$allocated" ]] || allocated="${SLURM_STEP_GPUS:-}"
  [[ -n "$allocated" ]] || allocated="${SLURM_JOB_GPUS:-}"

  # Slurm's device cgroup exports the allocation through CUDA_VISIBLE_DEVICES.
  # Preserve those IDs (including UUIDs) instead of accidentally exposing the
  # first N physical cards on a mixed node.
  if [[ -z "$visible" || "$visible" == "auto" ]]; then
    visible="$allocated"
  fi
  if [[ -z "$visible" || "$visible" == "auto" ]]; then
    local discovered=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      discovered="$(nvidia-smi -L 2>/dev/null | awk 'NF {count++} END {print count + 0}')"
    fi
    if [[ "${DRY_RUN:-0}" == "1" && ! "$discovered" =~ ^[1-9][0-9]*$ ]]; then
      discovered=8
    fi
    [[ "$discovered" =~ ^[1-9][0-9]*$ ]] ||
      fatal "GPU_IDS is unset and no allocated GPU list was discoverable"
    visible="$(seq -s, 0 $((discovered - 1)))"
  fi
  visible="${visible//[[:space:]]/}"
  local visible_count
  visible_count="$(_csv_count "$visible")" || fatal "invalid GPU_IDS: $visible"

  if [[ "${NUM_GPUS:-}" == "" || "${NUM_GPUS:-}" == "auto" ]]; then
    NUM_GPUS="$visible_count"
  fi
  [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || fatal "NUM_GPUS must be a positive integer: $NUM_GPUS"
  (( visible_count == NUM_GPUS )) ||
    fatal "GPU_IDS has $visible_count entries but NUM_GPUS=$NUM_GPUS"

  if [[ "${DP_SIZE:-}" == "" || "${DP_SIZE:-}" == "auto" ]]; then
    DP_SIZE="$NUM_GPUS"
  fi
  [[ "$DP_SIZE" =~ ^[1-9][0-9]*$ ]] || fatal "DP_SIZE must be a positive integer: $DP_SIZE"
  [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || fatal "TP_SIZE must be a positive integer: $TP_SIZE"
  (( DP_SIZE * TP_SIZE == NUM_GPUS )) ||
    fatal "DP_SIZE*TP_SIZE must equal NUM_GPUS (got ${DP_SIZE}*${TP_SIZE}!=$NUM_GPUS)"

  GPU_IDS="$visible"
  export GPU_IDS NUM_GPUS DP_SIZE TP_SIZE
}

resolve_training_accumulation() {
  [[ "$AUTO_SCALE_GRAD_ACCUM" == "1" ]] || return 0
  [[ "$TARGET_EFFECTIVE_BATCH" =~ ^[1-9][0-9]*$ ]] ||
    fatal "TARGET_EFFECTIVE_BATCH must be a positive integer: $TARGET_EFFECTIVE_BATCH"
  [[ "$PER_DEVICE_BATCH" =~ ^[1-9][0-9]*$ ]] ||
    fatal "PER_DEVICE_BATCH must be a positive integer: $PER_DEVICE_BATCH"
  local denominator=$((NUM_GPUS * PER_DEVICE_BATCH))
  local lower=$((TARGET_EFFECTIVE_BATCH / denominator))
  local upper=$((lower + 1))
  (( lower >= 1 )) || lower=1
  local lower_batch=$((lower * denominator))
  local upper_batch=$((upper * denominator))
  if (( TARGET_EFFECTIVE_BATCH - lower_batch <= upper_batch - TARGET_EFFECTIVE_BATCH )); then
    GRAD_ACCUM="$lower"
  else
    GRAD_ACCUM="$upper"
  fi
  ACTUAL_EFFECTIVE_BATCH=$((NUM_GPUS * PER_DEVICE_BATCH * GRAD_ACCUM))
  if (( ACTUAL_EFFECTIVE_BATCH != TARGET_EFFECTIVE_BATCH )); then
    echo "[train] target effective batch=$TARGET_EFFECTIVE_BATCH is approximated by $ACTUAL_EFFECTIVE_BATCH for $NUM_GPUS GPUs"
  fi
  export GRAD_ACCUM ACTUAL_EFFECTIVE_BATCH
}

validate_common() {
  resolve_runtime_gpu_config
  resolve_training_accumulation
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" && -x "$ACCELERATE" ]] || \
    fatal "required Python/accelerate executable missing"
  [[ -s "$MODEL/config.json" && -s "$MODEL/model.safetensors.index.json" && \
     -s "$MODEL/tokenizer_config.json" && -s "$MODEL/tokenizer.json" ]] || \
    fatal "invalid or incomplete model: $MODEL"
  "$PYTHON_BIN" - "$MODEL" <<'PY' || fatal "model shard validation failed: $MODEL"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("model index has no weight shards")
missing = [name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"missing or empty model shards: {missing}")
PY
  [[ -d "$MATH_DATA_ROOT" && -f "$SCRIPT_DIR/sas_pipeline.py" ]] || \
    fatal "MATH data or SAS runner missing"
  [[ "$GPU_IDS" =~ ^[^,]+(,[^,]+)*$ ]] || fatal "invalid GPU_IDS: $GPU_IDS"
  [[ "$DP_SIZE" =~ ^[1-9][0-9]*$ && "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || \
    fatal "DP_SIZE and TP_SIZE must be positive integers"
  local gpu_count
  gpu_count="$(_csv_count "$GPU_IDS")" || fatal "invalid GPU_IDS: $GPU_IDS"
  (( gpu_count == NUM_GPUS && DP_SIZE * TP_SIZE == NUM_GPUS )) || \
    fatal "GPU topology mismatch: ids=$gpu_count dp=$DP_SIZE tp=$TP_SIZE num=$NUM_GPUS"
  [[ "$ROLLOUT_MAX_NEW_TOKENS" == "8192" && "$EVAL_MAX_NEW_TOKENS" == "8192" ]] || \
    fatal "rollout/eval max tokens must match MATH JCA at 8192"
  [[ "$MAX_MODEL_LEN" == "40960" ]] || fatal "MAX_MODEL_LEN must match MATH JCA at 40960"
  [[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
  [[ "$EVAL_ENABLE_THINKING" == "0" || "$EVAL_ENABLE_THINKING" == "1" ]] || \
    fatal "EVAL_ENABLE_THINKING must be 0 or 1"
  [[ "$EVAL_REQUIRE_THINKING" == "0" || "$EVAL_REQUIRE_THINKING" == "1" ]] || \
    fatal "EVAL_REQUIRE_THINKING must be 0 or 1"
  [[ "$RUN_BASE_EVAL" == "0" || "$RUN_BASE_EVAL" == "1" ]] || \
    fatal "RUN_BASE_EVAL must be 0 or 1"
  [[ "$EVAL_ONLY" == "0" || "$EVAL_ONLY" == "1" ]] || \
    fatal "EVAL_ONLY must be 0 or 1"
  [[ "$SKIP_ROLLOUT_SCORE_IF_COMPLETE" == "0" || "$SKIP_ROLLOUT_SCORE_IF_COMPLETE" == "1" ]] || \
    fatal "SKIP_ROLLOUT_SCORE_IF_COMPLETE must be 0 or 1"
  [[ "$ALLOW_WORLD_SIZE_CHANGE_RESUME" == "0" || "$ALLOW_WORLD_SIZE_CHANGE_RESUME" == "1" ]] || \
    fatal "ALLOW_WORLD_SIZE_CHANGE_RESUME must be 0 or 1"
  [[ "$MAX_TRAIN_STEPS" =~ ^[0-9]+$ ]] || fatal "MAX_TRAIN_STEPS must be a non-negative integer"
  [[ "$TRAIN_SAVE_STEPS" =~ ^[0-9]+$ ]] || fatal "TRAIN_SAVE_STEPS must be a non-negative integer"
  [[ "$TRAIN_SAVE_TOTAL_LIMIT" =~ ^[1-9][0-9]*$ ]] || fatal "TRAIN_SAVE_TOTAL_LIMIT must be positive"
  [[ "$EVAL_REQUIRE_THINKING" == "0" || "$EVAL_ENABLE_THINKING" == "1" ]] || \
    fatal "EVAL_REQUIRE_THINKING=1 requires EVAL_ENABLE_THINKING=1"
  if [[ -n "$EVAL_PROBLEM_IDS_FILE" ]]; then
    [[ -f "$EVAL_PROBLEM_IDS_FILE" ]] || fatal "eval problem IDs file missing: $EVAL_PROBLEM_IDS_FILE"
    local eval_id_count
    eval_id_count="$($PYTHON_BIN - "$EVAL_PROBLEM_IDS_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ids = []
for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    value = json.loads(line)
    problem_id = str(value.get("problem_id", "")).strip() if isinstance(value, dict) else ""
    if not problem_id:
        raise SystemExit(f"missing problem_id at {path}:{line_number}")
    ids.append(problem_id)
if len(ids) != len(set(ids)):
    raise SystemExit(f"duplicate problem IDs in {path}")
print(len(ids))
PY
)" || fatal "invalid eval problem IDs file: $EVAL_PROBLEM_IDS_FILE"
    [[ "$EVAL_START" =~ ^[0-9]+$ && "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]] || \
      fatal "EVAL_START/EVAL_LIMIT must define a positive slice"
    (( EVAL_START + EVAL_LIMIT <= eval_id_count )) || \
      fatal "eval slice ${EVAL_START}+${EVAL_LIMIT} exceeds problem ID count $eval_id_count"
  fi
}

wait_idle_gpus() {
  [[ "$WAIT_FOR_IDLE_GPUS" == "1" ]] || return 0
  # Slurm's device cgroup already isolates an allocation.  A mixed node may
  # legitimately show unrelated jobs in nvidia-smi, so a global idle probe
  # would wait forever or race with another allocation.
  if [[ "${GPU_GUARD_USE_ALLOCATION:-1}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[gpu] using Slurm GPU allocation; skipping global idle probe"
    return 0
  fi
  gpu_guard_wait_idle "SAS" "$GPU_IDS" "$GPU_IDLE_MAX_MEMORY_MIB" \
    "$GPU_IDLE_CHECKS" "$GPU_POLL_SECONDS" "$GPU_IDLE_TIMEOUT_SECONDS" ||
    fatal "GPU readiness check failed"
}

SERVER_PID=""
SERVER_LOG=""
SERVER_RUNTIME_ROOT=""

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
    while kill -0 -- "-$SERVER_PID" 2>/dev/null || kill -0 "$SERVER_PID" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
        break
      fi
      sleep 1
    done
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [[ -n "$SERVER_RUNTIME_ROOT" && -d "$SERVER_RUNTIME_ROOT" && \
        "$(basename "$SERVER_RUNTIME_ROOT")" == jca_math_sas14b.* ]]; then
    rm -rf -- "$SERVER_RUNTIME_ROOT"
  fi
  SERVER_RUNTIME_ROOT=""
}

wait_server() {
  local port="$1" expected="$2" deadline=$((SECONDS + SERVER_WAIT_TIMEOUT)) response
  while true; do
    kill -0 "$SERVER_PID" 2>/dev/null || {
      tail -n 100 "$SERVER_LOG" >&2 || true
      fatal "vLLM exited before serving $expected"
    }
    if response="$(curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" && \
       [[ "$response" == *"\"${expected}\""* ]]; then
      echo "[server] ready model=$expected port=$port"
      return 0
    fi
    (( SECONDS < deadline )) || {
      tail -n 100 "$SERVER_LOG" >&2 || true
      fatal "timed out waiting for $expected"
    }
    sleep 5
  done
}

start_server() {
  local port="$1" served_name="$2" max_model_len="$3" log_path="$4"
  shift 4
  local runtime_root
  gpu_guard_wait_port_free "$served_name" "127.0.0.1" "$port" "$SERVER_WAIT_TIMEOUT" "$GPU_POLL_SECONDS" ||
    fatal "port 127.0.0.1:$port is not free before $served_name"
  runtime_root="$(mktemp -d "${TMPDIR:-/tmp}/jca_math_sas14b.XXXXXX")"
  SERVER_RUNTIME_ROOT="$runtime_root"
  mkdir -p "$runtime_root/ray" "$runtime_root/torchinductor" "$(dirname "$log_path")"
  SERVER_LOG="$log_path"
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1 --port "$port"
    --model "$MODEL" --served-model-name "$served_name"
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE"
    --dtype bfloat16 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$max_model_len" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --trust-remote-code --enforce-eager --enable-prefix-caching --enable-chunked-prefill
    "$@"
  )
  setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    TMPDIR="$runtime_root" RAY_TMPDIR="$runtime_root/ray" \
    TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor" \
    "${command[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  wait_server "$port" "$served_name"
}

run_pipeline() {
  env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$SCRIPT_DIR/sas_pipeline.py" "$@"
}
