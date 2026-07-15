#!/usr/bin/env bash
# H200 job worker: poll git → claim jobs/pending → run allowlisted command → tee logs.
# Start once in tmux:  tmux new -s pkworker './scripts/remote_worker.sh'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
cd "$ROOT"

POLL_SEC="${REMOTE_WORKER_POLL_SEC:-45}"
BRANCH="${REMOTE_WORKER_BRANCH:-main}"
PUSH_STATUS="${REMOTE_WORKER_PUSH_STATUS:-1}"
SCRATCH="${PRIORITYKV_SCRATCH:-$ROOT/../prioritykv}"
LOG_DIR="$SCRATCH/logs"
STATUS_DIR="$ROOT/jobs/status"

mkdir -p "$LOG_DIR" \
  "$ROOT/jobs/pending" "$ROOT/jobs/running" "$ROOT/jobs/done" \
  "$ROOT/jobs/failed" "$STATUS_DIR"

log() { printf '[remote_worker %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# Parse simple key: value YAML (no nested structures).
yaml_get() {
  local file="$1" key="$2" default="${3:-}"
  local line
  line="$(grep -E "^${key}:" "$file" 2>/dev/null | head -n1 || true)"
  if [[ -z "$line" ]]; then
    printf '%s' "$default"
    return
  fi
  line="${line#*:}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  # strip matching quotes
  if [[ "$line" =~ ^\"(.*)\"$ ]]; then
    line="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ ^\'(.*)\'$ ]]; then
    line="${BASH_REMATCH[1]}"
  fi
  printf '%s' "$line"
}

command_allowed() {
  local cmd="$1"
  # Allow: python scripts/foo.py …  OR  uv run python scripts/foo.py …
  if [[ "$cmd" =~ ^(uv[[:space:]]+run[[:space:]]+)?python[[:space:]]+scripts/[A-Za-z0-9_.-]+\.py([[:space:]].*)?$ ]]; then
    return 0
  fi
  return 1
}

sync_repo() {
  git fetch origin "$BRANCH" --quiet
  # Never reset --hard here: that would wipe claimed/archived job files.
  if ! git merge --ff-only "origin/$BRANCH" >/dev/null 2>&1; then
    log "WARN: ff-only merge failed (local job commits or diverged history); skipping pull this tick"
    return 1
  fi
}

write_status_files() {
  local job_id="$1" exit_code="$2" log_path="$3" job_yaml="$4"
  local finished
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local scratch_status="$LOG_DIR/${job_id}.status"
  cat >"$scratch_status" <<EOF
job_id=${job_id}
exit=${exit_code}
finished_at=${finished}
log=${log_path}
job_yaml=${job_yaml}
EOF
  cat >"$STATUS_DIR/${job_id}.json" <<EOF
{
  "job_id": "${job_id}",
  "exit": ${exit_code},
  "finished_at": "${finished}",
  "log": "${log_path}",
  "job_yaml": "${job_yaml}"
}
EOF
}

try_push_job_state() {
  local job_id="$1" dest_dir="$2"  # done or failed
  if [[ "$PUSH_STATUS" != "1" ]]; then
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi
  # Stage archive + status; ensure pending deletion is recorded.
  git add "jobs/${dest_dir}/${job_id}.yaml" "jobs/status/${job_id}.json" 2>/dev/null || true
  if [[ -f "jobs/pending/${job_id}.yaml" ]]; then
    git rm -f "jobs/pending/${job_id}.yaml" 2>/dev/null || true
  else
    git add -u "jobs/pending/${job_id}.yaml" 2>/dev/null || true
  fi
  if git diff --cached --quiet 2>/dev/null; then
    return 0
  fi
  if git commit -m "worker: ${dest_dir} ${job_id}" >/dev/null 2>&1; then
    if git push origin "HEAD:${BRANCH}" >/dev/null 2>&1; then
      log "pushed status for ${job_id} → ${dest_dir}"
    else
      log "WARN: push failed for ${job_id} (scratch status still written)"
    fi
  else
    log "WARN: commit failed for ${job_id}"
  fi
}

run_one_job() {
  local pending="$1"
  local base job_id command gpus sync_cuda timeout_sec
  base="$(basename "$pending")"
  job_id="$(yaml_get "$pending" id "${base%.yaml}")"
  command="$(yaml_get "$pending" command)"
  gpus="$(yaml_get "$pending" gpus "${CUDA_VISIBLE_DEVICES:-6,7}")"
  sync_cuda="$(yaml_get "$pending" sync_cuda false)"
  timeout_sec="$(yaml_get "$pending" timeout_sec 0)"

  if [[ -z "$command" ]]; then
    log "REJECT ${job_id}: missing command"
    mv "$pending" "$ROOT/jobs/failed/${job_id}.yaml"
    write_status_files "$job_id" 2 "$LOG_DIR/${job_id}.log" "jobs/failed/${job_id}.yaml"
    try_push_job_state "$job_id" failed
    return
  fi
  if ! command_allowed "$command"; then
    log "REJECT ${job_id}: command not allowlisted: ${command}"
    mv "$pending" "$ROOT/jobs/failed/${job_id}.yaml"
    write_status_files "$job_id" 2 "$LOG_DIR/${job_id}.log" "jobs/failed/${job_id}.yaml"
    echo "rejected: command not allowlisted" >"$LOG_DIR/${job_id}.log"
    try_push_job_state "$job_id" failed
    return
  fi

  local running="$ROOT/jobs/running/${job_id}.yaml"
  mv "$pending" "$running"

  local scratch_status="$LOG_DIR/${job_id}.status"
  if [[ -f "$scratch_status" ]]; then
    local prev_exit
    prev_exit="$(grep -E '^exit=' "$scratch_status" | head -n1 | cut -d= -f2-)"
    local dest=done
    [[ "${prev_exit:-1}" != "0" ]] && dest=failed
    log "SKIP ${job_id}: already finished (exit=${prev_exit:-?}); archiving → ${dest}"
    mv "$running" "$ROOT/jobs/${dest}/${job_id}.yaml"
    try_push_job_state "$job_id" "$dest"
    return
  fi

  if [[ "$sync_cuda" == "true" || "$sync_cuda" == "True" || "$sync_cuda" == "1" ]]; then
    log "sync_cuda for ${job_id}"
    "$ROOT/scripts/sync.sh" --cuda
  fi

  if [[ ! -d "$ROOT/.venv" ]]; then
    log "missing .venv — running ./scripts/sync.sh --cuda"
    "$ROOT/scripts/sync.sh" --cuda
  fi

  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  export CUDA_VISIBLE_DEVICES="$gpus"
  export PRIORITYKV_SCRATCH="$SCRATCH"
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  # H200 INT4 / torch JIT (HANDOFF_W3_INT4 §B): toolkit on PATH for nvcc.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  # vLLM V1 engine + prior torch CUDA init in same process → need spawn
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

  local log_path="$LOG_DIR/${job_id}.log"
  log "START ${job_id}: ${command} (gpus=${gpus} timeout=${timeout_sec})"
  {
    echo "=== remote_worker job=${job_id} started=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "command=${command}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "CUDA_HOME=${CUDA_HOME}"
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    echo "PRIORITYKV_SCRATCH=${PRIORITYKV_SCRATCH}"
    echo "cwd=${ROOT}"
  } >"$log_path"

  local exit_code=0
  set +e
  if [[ "${timeout_sec}" =~ ^[0-9]+$ ]] && [[ "${timeout_sec}" -gt 0 ]]; then
    # Prefer GNU timeout; fall back to bare run if missing.
    if command -v timeout >/dev/null 2>&1; then
      timeout --signal=TERM "${timeout_sec}" bash -lc "cd \"$ROOT\" && ${command}" \
        >>"$log_path" 2>&1
      exit_code=$?
    else
      bash -lc "cd \"$ROOT\" && ${command}" >>"$log_path" 2>&1
      exit_code=$?
    fi
  else
    bash -lc "cd \"$ROOT\" && ${command}" >>"$log_path" 2>&1
    exit_code=$?
  fi
  set -e

  echo "=== finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${exit_code} ===" >>"$log_path"

  local dest=done
  if [[ "$exit_code" -ne 0 ]]; then
    dest=failed
  fi
  mv "$running" "$ROOT/jobs/${dest}/${job_id}.yaml"
  write_status_files "$job_id" "$exit_code" "$log_path" "jobs/${dest}/${job_id}.yaml"
  try_push_job_state "$job_id" "$dest"
  log "END ${job_id}: exit=${exit_code} → jobs/${dest}/"
}

log "starting poll=${POLL_SEC}s branch=${BRANCH} scratch=${SCRATCH} push_status=${PUSH_STATUS}"

while true; do
  sync_repo || log "WARN: sync_repo failed"

  shopt -s nullglob
  pending_files=("$ROOT"/jobs/pending/*.yaml)
  shopt -u nullglob

  if [[ ${#pending_files[@]} -gt 0 ]]; then
    # One job at a time (shared 2-GPU cap).
    run_one_job "${pending_files[0]}"
  else
    log "idle"
  fi

  sleep "$POLL_SEC"
done
