# PriorityKV — blog draft (publish after arXiv)

**One-liner:** What you drop from the KV cache matters more than how hard you compress it — for agents.

## The problem

Long tool-using chats store schemas, superseding instructions, and IDs in the KV cache. Uniform eviction (sink+recent) can look fine on average metrics while silently breaking those agent behaviors.

## What we found

1. **Structure-aware keep** restores tool / supersession / multi-turn reliability at matched keep budgets where uniform keep collapses.
2. Soft **INT4 at 75%** does *not* create a PriorityBench quality gap vs FullKV (we checked; it’s falsified).
3. Systems value then comes from a **packed BF16/INT4** cache with FlashInfer-backed decode: packed bytes + honest H200 latency (e2e≈FullKV, TPOT ~1.2×), quality matched on a locked 240-example agent bench — with an honest cold-scratch caveat on peak VRAM.

## Reproduce

- Repo: https://github.com/Arush777/Priority_KV  
- Freeze: `FINAL_RUN_MANIFEST.yaml`  
- H200: see `docs/H200_SETUP.md` · worker `scripts/remote_worker.sh`

## arXiv

Link TBD after upload of `paper/prioritykv_arxiv_draft.md` (expand with FP8 / guardrails / Gemma job results).

## Cite the claim carefully

Do **not** say “INT4 hurts agent quality at 0.75.” Say eviction/missing-state hurts; structure keep helps; mixed dtype wins on bytes/latency at matched quality.
