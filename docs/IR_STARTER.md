# IR starter notes (seed for friend-led idea)

These are **starter sparks**, not locked decisions. Friend agent owns S1
(direction). Update `scopes/PROJECT_SCOPE.md` after ACK_SCOPE.

## Classic IR + LLM stack

1. **Corpus** — collection of documents/passages
2. **Query** — user question / search string
3. **Retriever** — return top-k relevant passages
4. **Metrics** — Recall@k, MRR, nDCG@k
5. **Optional RAG** — LLM answers grounded on retrieved passages

## Possible v0 angles (pick one)

| Angle | Why interesting |
|-------|-----------------|
| BM25 vs dense (e5/bge) on a small public set | Clean baseline story |
| Hybrid fusion (RRF) | Practical IR recipe |
| Query rewriting for retrieval | LLM-for-IR without full RAG |
| Failure analysis on hard queries | Research narrative |

## Suggested first deliverables

- `data/` small sample corpus + qrels
- `src/retrieve_bm25.py` baseline
- `src/eval.py` metrics
- Notebook or README with one measured number

Friend: CLAIM S1 and state the chosen angle in Telegram.
