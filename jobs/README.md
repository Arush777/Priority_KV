# H200 jobs — evidence archive + live queue

## Live queue (agent → H200, no SSH)

```
Agent box                              H200 pkworker (tmux)
─────────                              ────────────────────
1. edit code / add jobs/pending/<id>.yaml
2. git push  ─────────────────────────► 3. poll ~45s, ff-only pull
                                        4. run allowlisted python scripts/*.py
                                        5. write jobs/status + jobs/results
6. git pull / pull_job.sh ◄──────────── 7. git push status
```

| Dir | Purpose |
|---|---|
| `pending/` | New jobs (commit + push to enqueue) |
| `running/` | Claimed locally on H200 only (gitignored) |
| `done/` / `failed/` | Terminal outcomes |
| `status/` | Thin JSON: exit, decision, pass |
| `results/<id>/` | Debug bundle (logs, summary, nvidia-smi) |

**Helpers (agent box):** `./scripts/pull_job.sh [--watch] <id>`  
**Worker (H200 once):** `tmux new -s pkworker './scripts/remote_worker.sh'`  
**Unstick H200:** `bash scripts/h200_bootstrap_pkworker.sh`

### Job YAML

```yaml
id: diag_nvidia_smi_r7
command: python scripts/diag_nvidia_smi.py --out-tag r7
gpus: "0,1"          # max TWO ids
sync_cuda: false
timeout_sec: 120
```

Rules: `command` must be `python scripts/<name>.py …` · filename = `id` · **max 2 GPUs**.

## Evidence archive (paper freeze)

| Dir | Contents |
|---|---|
| `manifests/` | Canonical manuscript run commands |
| `results/<canonical_id>/` | Frozen result bundles cited in the paper |

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) and [`../docs/H200_QUEUE.md`](../docs/H200_QUEUE.md).
