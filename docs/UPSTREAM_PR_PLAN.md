# Upstream PR plan — PriorityKV → FlashInfer / Transformers

**Goal:** one focused PR, not a monorepo dump.

## Preferred target: FlashInfer

**Surface:** homogeneous page multicall + LSE merge already validated
(`w6e` PARITY_PASS, max abs ~4.88e-4). Our decode path:
`src/prioritykv/qwen3_fi_shim.py`, `fi_mixed_decode.py`, `packed_mixed_cache.py`.

**PR shape (draft):**

1. Minimal example: two-page BF16+INT4 (or BF16+BF16) decode attention via
   `flashinfer.merge_state` with documented LSE contract.
2. Short doc note: “PriorityKV agent-KV mixed-precision pages.”
3. Link to arXiv + repro once public.

**Do not** upstream: PriorityBench generators, job queue, Qwen-specific chat hacks.

## Fallback: Hugging Face Transformers

If FlashInfer PR is blocked: document a `DynamicCache`-compatible wrapper that
exposes packed pages + FI attend as an optional attention backend example
(gist or transformers examples/), still citing parity tests.

## Checklist before opening

- [ ] Isolate ≤3 files of production-quality example code
- [ ] CPU/GPU parity test snippet
- [ ] License headers clean
- [ ] Issue filed describing LSE contract footgun we hit (0.085 → 4.88e-4)

## Tracking

- Issue URL: TBD  
- PR URL: TBD  
- Date opened: TBD  
