gpu_guard_wait_idle() {
  local label="$1"
  local gpu_ids="$2"
  local max_memory_mib="$3"
  local required_checks="$4"
  local poll_seconds="$5"
  local timeout_seconds="$6"
  [[ "${WAIT_FOR_IDLE_GPUS:-1}" == 1 ]] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[gpu] nvidia-smi is unavailable before $label" >&2
    return 1
  }
  [[ "$max_memory_mib" =~ ^[0-9]+$ ]] || {
    echo "[gpu] invalid GPU_IDLE_MAX_MEMORY_MIB: $max_memory_mib" >&2
    return 1
  }
  [[ "$required_checks" =~ ^[1-9][0-9]*$ ]] || {
    echo "[gpu] invalid GPU_IDLE_CHECKS: $required_checks" >&2
    return 1
  }
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "[gpu] invalid GPU_POLL_SECONDS: $poll_seconds" >&2
    return 1
  }
  [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "[gpu] invalid GPU_IDLE_TIMEOUT_SECONDS: $timeout_seconds" >&2
    return 1
  }

  local expected_gpus
  expected_gpus="$(awk -F',' '{print NF}' <<<"${gpu_ids//[[:space:]]/}")"
  local checks=0 deadline=$((SECONDS + timeout_seconds))
  echo "[gpu] waiting for stable idle GPUs before $label (max_memory=${max_memory_mib}MiB)"
  while (( checks < required_checks )); do
    local gpu_rows app_rows row_count max_used compact
    if ! gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>&1)"; then
      checks=0
      echo "[gpu] nvidia-smi query failed before $label: ${gpu_rows//$'\n'/ }"
    else
      row_count="$(awk 'NF {count++} END {print count + 0}' <<<"$gpu_rows")"
      if (( row_count != expected_gpus )); then
        checks=0
        echo "[gpu] expected ${expected_gpus} GPU rows, got ${row_count}; waiting before $label"
      elif ! app_rows="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
        checks=0
        echo "[gpu] compute-process query failed before $label: ${app_rows//$'\n'/ }"
      else
        compact="$(tr -d '[:space:]' <<<"$app_rows")"
        [[ "$compact" == "Norunningprocessesfound" ]] && compact=""
        max_used="$(awk -F',' 'NF >= 2 {gsub(/[[:space:]]/, "", $2); if (($2 + 0) > max) max = $2 + 0} END {print max + 0}' <<<"$gpu_rows")"
        if [[ -z "$compact" ]] && (( max_used <= max_memory_mib )); then
          checks=$((checks + 1))
          echo "[$(date '+%F %T')] all GPUs idle (max_memory=${max_used}MiB, ${checks}/${required_checks})"
        else
          checks=0
          echo "[$(date '+%F %T')] GPUs occupied (max_memory=${max_used}MiB, pids=${compact:-none}); waiting before $label"
        fi
      fi
    fi
    (( checks >= required_checks )) && return 0
    (( SECONDS < deadline )) || {
      echo "[gpu] GPUs did not become reliably idle before $label" >&2
      return 1
    }
    sleep "$poll_seconds"
  done
}

gpu_guard_port_in_use() {
  local host="$1" port="$2"
  if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; then
    return 0
  fi
  if [[ "$host" == 127.0.0.1 || "$host" == localhost ]]; then
    if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
      exec 3>&-
      return 0
    fi
  fi
  return 1
}

gpu_guard_wait_port_free() {
  local label="$1" host="$2" port="$3" timeout_seconds="$4" poll_seconds="$5"
  local deadline=$((SECONDS + timeout_seconds))
  while gpu_guard_port_in_use "$host" "$port"; do
    echo "[port] ${host}:${port} is still occupied before ${label}; waiting"
    (( SECONDS < deadline )) || {
      echo "[port] ${host}:${port} did not become free before ${label}" >&2
      return 1
    }
    sleep "$poll_seconds"
  done
  return 0
}

gpu_guard_pid_command() {
  local pid="$1"
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null
}

gpu_guard_pid_owned() {
  local pid="$1" project_root="$2" owner_token="$3" allow_vllm="${4:-0}"
  local command environment
  command="$(gpu_guard_pid_command "$pid" 2>/dev/null || true)"
  [[ -n "$command" ]] || return 1
  if [[ "$command" == *"$project_root/baseline/MATH/"* ||
        "$command" == *"$project_root/scripts/rl_train"* ||
        "$command" == *"sas_pipeline.py"* ||
        "$command" == *"rl_train_pairwise_rwr.py"* ||
        ("$allow_vllm" == 1 && "$command" == *"vllm.entrypoints.openai.api_server"*) ]]; then
    return 0
  fi
  if [[ -n "$owner_token" && -r "/proc/${pid}/environ" ]]; then
    environment="$(tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null || true)"
    grep -Fqx "JCA_MATH_OWNER=${owner_token}" <<<"$environment" && return 0
  fi
  return 1
}

gpu_guard_cleanup_math() {
  local label="$1"
  local project_root="$2"
  local run_root="$3"
  local owner_token="$4"
  local ports_csv="$5"
  local grace_seconds="${6:-15}"
  local timeout_seconds="${7:-120}"
  local poll_seconds="${8:-5}"
  local pid_file pid port command pgid sid deadline remaining
  local -a kill_pids=()
  local -a cleanup_ports=()
  declare -A candidates=()
  declare -A port_candidates=()
  declare -A pid_file_candidates=()

  [[ -n "$project_root" && -n "$run_root" ]] || {
    echo "[cleanup] invalid project/run root before ${label}" >&2
    return 1
  }
  [[ "$grace_seconds" =~ ^[1-9][0-9]*$ &&
     "$timeout_seconds" =~ ^[1-9][0-9]*$ &&
     "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "[cleanup] invalid cleanup timeout before ${label}" >&2
    return 1
  }

  while IFS= read -r pid_file; do
    [[ -f "$pid_file" ]] || continue
    pid="$(tr -dc '0-9' <"$pid_file" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
      candidates["$pid"]=1
      pid_file_candidates["$pid"]=1
    fi
  done < <(find "$run_root" -maxdepth 3 -type f -name '*.pid' -size -32c 2>/dev/null || true)

  IFS=',' read -r -a cleanup_ports <<<"$ports_csv"
  for port in "${cleanup_ports[@]}"; do
    port="${port//[[:space:]]/}"
    [[ "$port" =~ ^[1-9][0-9]*$ ]] || continue
    if command -v ss >/dev/null 2>&1; then
      while IFS= read -r pid; do
        if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
          candidates["$pid"]=1
          port_candidates["$pid"]=1
        fi
      done < <(
        ss -H -ltnp "sport = :${port}" 2>/dev/null |
          grep -oE 'pid=[0-9]+' | cut -d= -f2 || true
      )
    fi
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r pid; do
      [[ "$pid" =~ ^[1-9][0-9]*$ ]] && candidates["$pid"]=1
    done < <(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
        awk '$1 ~ /^[0-9]+$/ {print $1}' || true
    )
  else
    echo "[cleanup] nvidia-smi unavailable before ${label}; using PID/port cleanup only" >&2
  fi

  for pid in "${!candidates[@]}"; do
    kill -0 "$pid" 2>/dev/null || continue
    local allow_vllm=0
    [[ -n "${port_candidates[$pid]+present}" || -n "${pid_file_candidates[$pid]+present}" ]] && allow_vllm=1
    if gpu_guard_pid_owned "$pid" "$project_root" "$owner_token" "$allow_vllm"; then
      command="$(gpu_guard_pid_command "$pid" 2>/dev/null || true)"
      echo "[cleanup] terminating pid=${pid}: ${command:0:240}"
      kill_pids+=("$pid")
    else
      command="$(gpu_guard_pid_command "$pid" 2>/dev/null || echo '<unreadable>')"
      echo "[cleanup] preserving unrelated pid=${pid}: ${command:0:240}"
    fi
  done

  for pid in "${kill_pids[@]}"; do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ "$pgid" == "$pid" && "$sid" == "$pid" ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
    kill -TERM "$pid" 2>/dev/null || true
  done

  ((${#kill_pids[@]} == 0)) || sleep "$grace_seconds"

  deadline=$((SECONDS + timeout_seconds))
  while :; do
    remaining=0
    for pid in "${kill_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && remaining=$((remaining + 1))
    done
    for port in "${cleanup_ports[@]}"; do
      port="${port//[[:space:]]/}"
      [[ "$port" =~ ^[1-9][0-9]*$ ]] || continue
      gpu_guard_port_in_use 127.0.0.1 "$port" && remaining=$((remaining + 1))
    done
    (( remaining == 0 )) && {
      echo "[cleanup] ${label}: owned processes and known ports are clear"
      return 0
    }
    (( SECONDS >= deadline )) && break
    sleep "$poll_seconds"
  done

  for pid in "${kill_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || continue
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ "$pgid" == "$pid" && "$sid" == "$pid" ]]; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    kill -KILL "$pid" 2>/dev/null || true
  done
  echo "[cleanup] ${label}: timeout; some owned processes or ports remain" >&2
  return 1
}
