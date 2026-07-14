# Project scope — Priority_KV

Living scope for **Priority_KV** (PriorityKV-Agent). Agents may propose
changes; they must not silently expand scope unless a human posts `ACK_SCOPE`
in Telegram (see `COLLAB.md`).

**Repo:** https://github.com/Arush777/Priority_KV

## Goal

Build the PriorityKV-Agent artifact described in
`docs/PRIORITYKV_IMPLEMENTATION_PLAN.md`: mixed-precision paged KV cache
(BF16/INT4) that protects structurally critical pages so long multi-turn agent
traces do not silently lose tool schemas / instruction hierarchy under
compression.

Friend's agent leads the concrete research/eval angle (Workstream A /
PriorityBench). Arush's agent collaborates on systems/infra/collab bridge and
shared scaffolding.

## In scope (v0 bootstrap)

- Collaboration bridge (`collab_bridge`) + Telegram protocol
- Track the locked plan in `docs/PRIORITYKV_IMPLEMENTATION_PLAN.md`
- Scaffold repo layout for Workstream A (eval) and B (systems) as tasks land
- Small neutral utilities only when they unblock S1

## Out of scope (v0)

- Expanding beyond plan without ACK_SCOPE
- Auto-merge to `main` without PR
- Jobs on the partner's cluster account
- Paid APIs without human OK in Telegram

## Current workstreams

| ID | Owner | Status | Notes |
|----|-------|--------|-------|
| S0 | arush | active | Collab bridge + repo hygiene |
| S1 | friend | proposed | Lead PriorityBench / research angle from plan |
| S2 | arush | pending | Systems scaffolding once S1 choices land |

## Change log

- 2026-07-14: Retargeted to Priority_KV; ingested friend implementation plan v2.
