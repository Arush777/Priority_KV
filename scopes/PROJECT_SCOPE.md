# Project scope — Information Retrieval

Living scope for **Priority_KV**. Agents may propose changes; they must
not silently expand scope unless a human posts `ACK_SCOPE` in Telegram (see
`COLLAB.md`).

## Goal

Build a research / prototype project in **Information Retrieval (IR)** — the
classic LLM topic: retrieve relevant documents/passages for a query, optionally
augment generation (RAG), and measure quality.

Friend's agent proposes the concrete idea direction; Arush's agent collaborates
on implementation, experiments, and infra.

## In scope (v0)

- Problem statement + literature notes (sparse + dense retrieval)
- Dataset selection (public IR benchmarks or a small custom corpus)
- Baseline retriever (BM25 and/or dense embeddings)
- Evaluation harness (Recall@k, nDCG@k, MRR)
- Minimal RAG demo path (retrieve → optional LLM generate)
- Reproducible scripts / configs
- Collaboration via `collab_bridge` (Telegram + hourly ticks)

## Out of scope (v0)

- Production serving / multi-tenant product
- Training giant foundation models from scratch
- Paying external APIs without explicit human approval in Telegram
- Auto-merging to `main` without a PR review signal
- Cluster GPU jobs on the partner's account

## Current workstreams

| ID | Owner | Status | Notes |
|----|-------|--------|-------|
| S0 | both | active | Bootstrap bridge + repo hygiene |
| S1 | friend | proposed | Concrete IR idea / research angle (friend agent leads) |
| S2 | arush | pending | Implementation scaffolding once S1 lands |

## Change log

- 2026-07-14: Initial scope seeded for collab-bridge bootstrap.
