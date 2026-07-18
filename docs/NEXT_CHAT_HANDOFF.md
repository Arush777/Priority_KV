# Handoff — next chat (DeepMind packaging track)

**Date:** 2026-07-19 · **Repo:** `github.com:Arush777/Priority_KV` · **Branch:** `main`  
**Tip at handoff:** pull latest (`RESULTS.md` + science-core freeze should be on `main`)  
**Owner git identity:** `Arush777 <153831754+Arush777@users.noreply.github.com>`

## Read first (in order)

1. [`RESULTS.md`](../RESULTS.md) — point of project + canonical metrics  
2. [`docs/DATASET.md`](DATASET.md) — PriorityBench tasks (first principles)  
3. [`FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml) — freeze `SCIENCE_CORE_HOME_2026_07_19`  
4. [`docs/D3_CLOSE.md`](D3_CLOSE.md) — D3 closed; cold-scratch caveat  
5. [`paper/prioritykv_arxiv_draft.md`](../paper/prioritykv_arxiv_draft.md) — draft to finish  

Optional: `docs/decisions.md` (narrative), `docs/H200_SETUP.md` (ops), `jobs/README.md` (queue).

## Science core status: DONE — do not reopen casually

| Done | Do not |
|---|---|
| Reliability + packed systems + lock-240 + latency/peak | Soft-INT4 quality hunts |
| Publish GPU: FP8 PASS, guardrails PASS, Gemma r6 PASS | LongBench / full RULER matrices |
| D3 CLOSED (packed + FI shim) | Claim peak VRAM ≪ FullKV |

**Hard rules:** max **2 GPUs** on H200; no agents on `dgre2`; scratch unconstrained.

## Dataset (one paragraph)

**PriorityBench-A** = 240 locked multi-turn chats (8k/16k/32k). Three tasks × 80:
(1) **tool_schema** — valid tool JSON after long filler; (2) **instruction_supersession** —
obey latest constraint; (3) **multi_turn_state** — reuse early ID/path/pref. Scorers are
deterministic 0/1. Eviction experiments ask: does uniform drop kill these while structure keep saves them?

## Your job next (priority order)

1. **Paper** — polish `paper/prioritykv_arxiv_draft.md`; pull numeric tables from
   `FINAL_RUN_MANIFEST.yaml` / `jobs/status/` / H200 scratch JSONs if needed
   (`./scripts/fetch_results.sh` or force-push salvage). Aim arXiv tech-report submit.
2. **Figures** — keep_frac bar charts; lock-240 by length; latency pack/cold/e2e/TPOT;
   payload vs peak honesty plot.
3. **Blog** — ship `docs/BLOG_DRAFT.md` (or cut down to RESULTS narrative).
4. **Upstream PR** — follow `docs/UPSTREAM_PR_PLAN.md` (FlashInfer / packed shim slice).
5. **Outreach** — fill `docs/outreach_log.md` (real sends).
6. **Optional post-core systems** — page-streamed INT4 attend *without* full-layer
   BF16 cold scratch (explicitly out of science freeze).

## Ops cheatsheet

```bash
# Agent box
cd /u/arushh/Arush/Priority_KV
git pull --ff-only origin main
./scripts/pull_job.sh <job_id>

# H200 (human SSH)
cd /data/anupam/scratch/Priority_KV
git fetch && git reset --hard origin/main   # if ff-only stuck
tmux new -d -s pkworker './scripts/remote_worker.sh'
```

Pending queue should be **empty**. Failed `pub_c_*` r2/r3/r5 are audit only.

## Claim wording (copy-paste)

> Uniform KV eviction/missing-state silently corrupts tool schemas and instruction
> hierarchies in long agent traces. Structure-aware retention preserves them at matched
> keep budgets. Soft INT4 at int4_frac=0.75 does not open a PriorityBench quality gap.
> Systems value = packed payload bytes + honest H200 latency, with cold-scratch peak caveat.

## Success for this packaging phase

- [ ] arXiv draft ready to submit (or submitted)  
- [ ] Blog published or scheduled  
- [ ] Upstream PR opened  
- [ ] ≥1 real outreach logged  
- [ ] No new GPU science unless user explicitly reopens scope  
