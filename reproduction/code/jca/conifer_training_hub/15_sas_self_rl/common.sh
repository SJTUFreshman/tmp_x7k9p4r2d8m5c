#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export SCRIPT_DIR
set -a
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"
set +a

fatal() { echo "[fatal] $*" >&2; exit 1; }

jsonl_complete() {
  local output="$1" data="$2" start="$3" limit="$4" num_rollouts="$5"
  [[ -s "$output" && -s "$data" ]] || return 1
  "$PYTHON_BIN" - "$output" "$data" "$start" "$limit" "$num_rollouts" <<'PY'
import json
import sys

output_path, data_path, raw_start, raw_limit, raw_rollouts = sys.argv[1:]
start, limit, num_rollouts = int(raw_start), int(raw_limit), int(raw_rollouts)
with open(data_path, encoding="utf-8") as handle:
    data = [json.loads(line) for line in handle if line.strip()]
selected = data[start : start + limit if limit else None]
expected = {(str(row["problem_id"]), index) for row in selected for index in range(num_rollouts)}
actual = set()
with open(output_path, encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("problem_id")), int(row.get("rollout_idx", -1)))
        if key in actual:
            raise SystemExit(f"duplicate rollout key at {output_path}:{line_number}: {key}")
        actual.add(key)
if actual != expected:
    raise SystemExit(f"incomplete JSONL {output_path}: expected={len(expected)} actual={len(actual)}")
PY
}

reject_existing() {
  local path
  for path in "$@"; do
    [[ ! -e "$path" ]] || fatal "output already exists with RESUME=0; choose a new RUN_ID or enable resume: $path"
  done
}

validate_common() {
  local gpu_count
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" && -x "$ACCELERATE" ]] || fatal "required Python executable missing"
  [[ -s "$MODEL_8B/config.json" && -s "$MODEL_14B/config.json" ]] || fatal "8B or 14B model is missing"
  [[ -s "$TRAIN_DATA" && -s "$TEST_DATA" && -f "$SCRIPT_DIR/sas_conifer_pipeline.py" ]] || fatal "Conifer data or SAS runner missing"
  [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fatal "GPU_IDS must be comma separated"
  gpu_count="$(awk -F',' '{print NF}' <<<"$GPU_IDS")"
  [[ "$NUM_GPUS" == "$gpu_count" ]] || fatal "NUM_GPUS must equal the number of GPU_IDS"
  for value in RESUME DRY_RUN MOCK SKIP_3X8B; do
    [[ "${!value}" == 0 || "${!value}" == 1 ]] || fatal "$value must be 0 or 1"
  done
}

SERVER_PIDS=()
SERVER_LOGS=()

stop_servers() {
  local pid deadline
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
  while (( SECONDS < deadline )); do
    local alive=0
    for pid in "${SERVER_PIDS[@]:-}"; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
  SERVER_LOGS=()
}

check_ports_free() {
  "$PYTHON_BIN" - "$@" <<'PY'
import socket
import sys
for raw in sys.argv[1:]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(raw)))
PY
}

wait_server() {
  local port="$1" expected="$2" pid="$3" log="$4" deadline=$((SECONDS + SERVER_WAIT_TIMEOUT)) response
  while true; do
    kill -0 "$pid" 2>/dev/null || { tail -n 100 "$log" >&2 || true; fatal "vLLM exited before serving $expected"; }
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" && [[ "$response" == *"\"${expected}\""* ]]; then
      echo "[server] ready model=$expected port=$port"
      return 0
    fi
    (( SECONDS < deadline )) || { tail -n 100 "$log" >&2 || true; fatal "timed out waiting for $expected"; }
    sleep 3
  done
}

start_replicas() {
  local model="$1" served_name="$2" base_port="$3" log_prefix="$4" adapter="${5:-}"
  local base_name="$served_name"
  [[ -z "$adapter" ]] || base_name="${served_name}_base"
  local -a gpu_list=() ports=()
  IFS=',' read -r -a gpu_list <<<"$GPU_IDS"
  local rank gpu port log
  for rank in "${!gpu_list[@]}"; do ports+=("$((base_port + rank))"); done
  check_ports_free "${ports[@]}" || fatal "one or more replica ports are busy: ${ports[*]}"
  SERVER_PIDS=()
  SERVER_LOGS=()
  for rank in "${!gpu_list[@]}"; do
    gpu="${gpu_list[rank]//[[:space:]]/}"
    port=$((base_port + rank))
    log="${log_prefix}_replica_${rank}.log"
    local -a command=(
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
      --host 127.0.0.1 --port "$port" --model "$model"
      --served-model-name "$base_name" --tensor-parallel-size 1
      --dtype bfloat16 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --trust-remote-code
      --enable-prefix-caching --enable-chunked-prefill --async-scheduling
      --performance-mode "$VLLM_PERFORMANCE_MODE" --disable-uvicorn-access-log
      --default-chat-template-kwargs '{"enable_thinking":false}'
    )
    if [[ -n "$adapter" ]]; then
      command+=(--enable-lora --max-lora-rank "$LORA_RANK" --max-loras 1 --max-cpu-loras 1 --lora-dtype auto --lora-modules "${served_name}=${adapter}")
    fi
    setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      OMP_NUM_THREADS=4 LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${command[@]}" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
    SERVER_LOGS+=("$log")
  done
  for rank in "${!gpu_list[@]}"; do
    wait_server "$((base_port + rank))" "$served_name" "${SERVER_PIDS[rank]}" "${SERVER_LOGS[rank]}"
  done
}

endpoint_pool() {
  local base_port="$1" count result="" rank
  count="$(awk -F',' '{print NF}' <<<"$GPU_IDS")"
  for ((rank=0; rank<count; rank++)); do
    [[ -z "$result" ]] || result+=","
    result+="http://127.0.0.1:$((base_port + rank))/v1"
  done
  printf '%s\n' "$result"
}

run_pipeline() {
  env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$SCRIPT_DIR/sas_conifer_pipeline.py" "$@"
}
