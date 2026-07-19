# Serial run-to-P3 (one pkworker, one GPU)

## Rule
- **One** `pkworker` only. Kill extras: `tmux kill-session -t pkworker3` (etc.).
- **One job at a time.** Next job is enqueued only after the previous lands on git.
- Prefer a single empty H200 (`gpus: "1"` or `"3"`… — never 0/2 while busy). Cap = 1 GPU unless council says otherwise (max 2).

## Queue order
1. `p1_attn_baselines_s1_kf25_*` — running / next
2. `p1_attn_baselines_s2_kf25_*` — after s1
3. Aggregate P1 n=120 → council Fable+Codex
4. **P2** — FI-FullKV / streamed cold attend (latency claim); council before enqueue
5. **P3** — Llama-3.1 transfer + harden distractors; council before enqueue
6. Optional P0 matrix shards (kf/page/buried) if council says still needed

## On error
Worker pushes failed status → agent pulls → **Fable + Codex** → fix → push → requeue on empty GPU → continue.
