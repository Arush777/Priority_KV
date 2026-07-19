#!/usr/bin/env bash
# H200: force-publish scratch job status/results that failed to git-push.
# Usage (on dgre2):
#   bash scripts/h200_force_push_scratch_jobs.sh pub_a_d4_fp8_compare_gpu01_r1 pub_b_guardrails_gpu5_r1 pub_c_gemma_reduced_gpu5_r1
set -euo pipefail
cd /data/anupam/scratch/Priority_KV
SCRATCH="${PRIORITYKV_SCRATCH:-/data/anupam/scratch/prioritykv}"
IDS=("$@")
if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "usage: $0 <job_id> [job_id...]" >&2
  exit 2
fi

tmux kill-session -t pkworker 2>/dev/null || true
git fetch origin
git reset --hard origin/main

for job_id in "${IDS[@]}"; do
  st="$SCRATCH/logs/${job_id}.status"
  lg="$SCRATCH/logs/${job_id}.log"
  if [[ ! -f "$st" ]]; then
    echo "MISSING status: $st" >&2
    continue
  fi
  exit_code="$(grep -E '^exit=' "$st" | head -n1 | cut -d= -f2-)"
  dest=done
  [[ "${exit_code:-1}" != "0" ]] && dest=failed
  mkdir -p "jobs/${dest}" "jobs/status" "jobs/results/${job_id}"
  if [[ ! -f "jobs/${dest}/${job_id}.yaml" ]]; then
    cat >"jobs/${dest}/${job_id}.yaml" <<EOF
id: ${job_id}
command: python scripts/placeholder_recovered.py
# recovered from scratch ${st}
EOF
  fi
  # Prefer existing summary in results; else copy tail from log.
  if [[ -f "$lg" ]]; then
    tail -c 524288 "$lg" >"jobs/results/${job_id}/log_tail.txt" || true
    grep -E 'decision=|^out=|"decision"' "$lg" | tail -n 40 >"jobs/results/${job_id}/decision_lines.txt" || true
    out_path="$(grep -E '^out=' "$lg" | tail -n1 | sed 's/^out=//' || true)"
    if [[ -n "${out_path:-}" && -f "$out_path" ]]; then
      cp -f "$out_path" "jobs/results/${job_id}/summary.json" || true
    fi
  fi
  decision="$(grep -E '^decision=' "$st" | head -n1 | cut -d= -f2- || true)"
  pass_raw="$(grep -E '^pass=' "$st" | head -n1 | cut -d= -f2- || true)"
  finished="$(grep -E '^finished_at=' "$st" | head -n1 | cut -d= -f2- || true)"
  python3 - "$job_id" "$exit_code" "$finished" "$lg" "$dest" "$decision" "$pass_raw" <<'PY'
import json, sys
job_id, exit_code, finished, log, dest, decision, pass_raw = sys.argv[1:8]
pass_out = None
if pass_raw in ("True", "true"): pass_out = True
elif pass_raw in ("False", "false"): pass_out = False
elif pass_raw not in ("",): pass_out = pass_raw
obj = {
  "job_id": job_id,
  "exit": int(exit_code or 1),
  "finished_at": finished or None,
  "log": log,
  "job_yaml": f"jobs/{dest}/{job_id}.yaml",
  "decision": decision or None,
  "pass": pass_out,
  "results_dir": f"jobs/results/{job_id}",
  "note": "force-pushed from H200 scratch via h200_force_push_scratch_jobs.sh",
}
open(f"jobs/status/{job_id}.json", "w").write(json.dumps(obj, indent=2) + "\n")
print("wrote", f"jobs/status/{job_id}.json", obj.get("decision"))
PY
  git add "jobs/${dest}/${job_id}.yaml" "jobs/status/${job_id}.json" "jobs/results/${job_id}"
  git rm -f "jobs/pending/${job_id}.yaml" 2>/dev/null || true
done

if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git -c user.name="Arush777" -c user.email="153831754+Arush777@users.noreply.github.com" \
    commit -m "worker: salvage scratch status/results for publish jobs"
  git push origin HEAD:main
  echo "pushed OK"
fi

tmux new -d -s pkworker './scripts/remote_worker.sh'
tmux capture-pane -t pkworker -p | tail -20
ls jobs/pending/ || true
