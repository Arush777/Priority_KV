# H200 git queue pipeline (no SSH from agent)

Control GPU work from the agent box by **pushing jobs to GitHub**. H200 runs
`pkworker`, which polls `main`, executes allowlisted scripts, and pushes
`jobs/status` + `jobs/results` back.

## Architecture

```
┌─────────────────────────┐         git push          ┌──────────────────────────┐
│ Agent / Cursor (CCC)    │ ─────────────────────────►│ github.com/Arush777/…    │
│                         │                           └────────────┬─────────────┘
│ • edit code             │                                        │
│ • jobs/pending/*.yaml   │◄──────── git fetch/status ─────────────┤
│ • pull_job.sh / pkmon   │                                        │
└─────────────────────────┘                           ┌────────────▼─────────────┐
                                                      │ H200 dgre2  pkworker     │
                                                      │ poll 45s → run → push    │
                                                      │ scratch logs under       │
                                                      │ /data/anupam/scratch/…   │
                                                      └──────────────────────────┘
```

## One-time on H200 (you SSH once)

```bash
cd /data/anupam/scratch/Priority_KV
git fetch origin && git reset --hard origin/main
bash scripts/h200_bootstrap_pkworker.sh   # starts pkworker0 + pkworker1 (GPUs 0,1)
# Override: PKWORKER_GPUS="0 7" bash scripts/h200_bootstrap_pkworker.sh
tmux ls | grep pkworker
tmux capture-pane -t pkworker0 -p | tail -20
```

Each worker only claims jobs whose `gpus:` field equals its filter. Enqueue two
1-GPU jobs on disjoint empty GPUs for parallelism (hard cap: 2 GPUs total).

If you see `ff-only merge failed` forever: re-run bootstrap (`reset --hard origin/main`).

## Agent box — enqueue a run

```bash
cd /u/arushh/Arush/Priority_KV
# 1) write jobs/pending/<id>.yaml  (gpus ≤ 2)
# 2) commit + push as Arush777
git -c user.name="Arush777" -c user.email="153831754+Arush777@users.noreply.github.com" \
  commit -m "Enqueue <id>" && git push origin HEAD

# 3) watch (no SSH)
./scripts/pull_job.sh --watch <id>
```

## Agent box — monitors (always-on)

| Session | Role |
|---|---|
| `pkmon-poll` | Every 5 min: `git fetch` + `pull_job` for watched IDs |
| `pkmon` / Cursor loop | Wake every few min to diagnose failures + requeue |

Start poller:

```bash
tmux new -d -s pkmon-poll 'cd /u/arushh/Arush/Priority_KV && while true; do
  date -u; git fetch origin main && git merge --ff-only origin/main || true;
  ls jobs/pending jobs/done jobs/failed 2>/dev/null | head -40;
  for f in jobs/pending/*.yaml; do
    [ -f "$f" ] || continue;
    id=$(basename "$f" .yaml);
    ./scripts/pull_job.sh "$id" || true;
  done;
  sleep 300;
done'
```

## Hard rules

- **Max 2 GPUs** per job (`gpus: "0,1"` or `"5,6"` …). **Prefer 1 GPU** whenever the
  workload fits (set `gpus: "7"` or a single free index from the latest diag).
- Never put secrets in job YAML; `HF_TOKEN` stays in H200 `.env`.
- Allowlisted commands only: `python scripts/<name>.py …`.
- Disk/scratch unconstrained under `$PRIORITYKV_SCRATCH`.

## Sanity probe (free GPUs)

Enqueue `diag_nvidia_smi_*` — worker captures `nvidia-smi` into `jobs/results/<id>/`.
After status push, agent reads that file via `pull_job.sh` (still no SSH).
Latest free-GPU snapshot (user, 2026-07-19): **0, 1, 7 empty**; 2–6 busy.
Prefer `gpus: "0"` / `"1"` with dual workers; leave 7 as spare.