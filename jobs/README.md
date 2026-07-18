# H200 job queue

Queued GPU/CPU jobs for the H200 worker (`scripts/remote_worker.sh`).
Agents never run on the H200 — only this queue + git + `uv`.

## Control everything from the agent box

```
Agent / Cursor                          H200 pkworker
─────────────────                       ─────────────────
1. code + pytest
2. jobs/pending/<id>.yaml
3. git push  ─────────────────────────► 4. poll (~45s), pull, run
                                        5. nvidia-smi before/after
                                        6. copy summary + log_tail
7. git pull / ./scripts/pull_job.sh ◄── 8. git push status+results
```

No SSH needed after **one** H200 worker restart onto this script.

## Layout

| Dir | Purpose |
|---|---|
| `pending/` | New jobs (commit + push to enqueue) |
| `running/` | Claimed by worker (gitignored; local only) |
| `done/` | Finished with exit 0 |
| `failed/` | Finished with non-zero exit |
| `status/` | JSON: `exit`, `decision`, `pass`, `results_dir` |
| `results/<id>/` | `summary.json`, `log_tail.txt`, `nvidia_smi*.txt`, `meta.json` |

## Job YAML schema

```yaml
id: w3_int4_assert_r1
command: python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only
gpus: "6,7"          # optional; default from .env
sync_cuda: false     # true only when lockfile / gpu deps changed
timeout_sec: 7200    # 0 = no timeout
```

Rules:

- `command` must be `python scripts/<name>.py …` or `uv run python scripts/<name>.py …`
- One job per file; filename should match `id`
- Do **not** put secrets in job files
- Scripts should print `out=/path/to/result.json` so the worker can ship `jobs/results/<id>/summary.json`

## Agent loop (no H200 laptop)

1. Implement + `uv run pytest` locally
2. Add `jobs/pending/<id>.yaml`
3. Commit + push (**author Arush777**)
4. Wait / poll: `./scripts/pull_job.sh --watch <id>`
5. Read `jobs/status/<id>.json` + `jobs/results/<id>/`

GPU snapshot anytime:

```yaml
# jobs/pending/diag_nvidia_smi_r1.yaml
id: diag_nvidia_smi_r1
command: python scripts/diag_nvidia_smi.py --out-tag r1
gpus: "6,7"
timeout_sec: 120
```

## Worker knobs (H200 `.env`)

```bash
REMOTE_WORKER_POLL_SEC=45
REMOTE_WORKER_BRANCH=main
REMOTE_WORKER_PUSH_STATUS=1
REMOTE_WORKER_PUSH_RESULTS=1   # ship jobs/results/* to git
REMOTE_WORKER_LOG_TAIL_BYTES=65536
```

## One-time H200 install (do while you have SSH)

```bash
cd /data/anupam/scratch/Priority_KV
tmux kill-session -t pkworker 2>/dev/null || true
git fetch origin
git reset --hard origin/main
tmux new -d -s pkworker './scripts/remote_worker.sh'
tmux ls
# optional check:
tmux capture-pane -t pkworker -p | tail -20
```

After that, leave H200 alone — control from git only.
