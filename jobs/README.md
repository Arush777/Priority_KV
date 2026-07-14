# H200 job queue

Queued GPU/CPU jobs for the H200 worker (`scripts/remote_worker.sh`).
Agents never run on the H200 — only this queue + git + `uv`.

## Layout

| Dir | Purpose |
|---|---|
| `pending/` | New jobs (commit + push to enqueue) |
| `running/` | Claimed by worker (gitignored; local only) |
| `done/` | Finished with exit 0 |
| `failed/` | Finished with non-zero exit |
| `status/` | Tiny JSON summaries written by the worker |

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
- One job per file; filename should match `id` (e.g. `w3_int4_assert_r1.yaml`)
- Do **not** put secrets in job files

## Agent loop

1. Implement + `uv run pytest` locally
2. Add `jobs/pending/<id>.yaml`
3. Commit and push
4. On H200, worker polls → pulls → runs → tees logs
5. Locally: `./scripts/fetch_results.sh`
6. Read `scratch_mirror/logs/<id>.log` + run JSON under `scratch_mirror/runs/`

## Logs / artifacts (not in git)

On H200 under `$PRIORITYKV_SCRATCH`:

- `logs/<id>.log` — full stdout/stderr
- `logs/<id>.status` — `exit=`, `finished_at=`, paths
- `runs/…` — experiment JSON (rsync via `fetch_results.sh`)
