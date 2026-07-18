# D3 CLOSE — Mixed paged BF16/INT4 backend

**Status:** **CLOSED** 2026-07-19  
**Freeze:** science-core home with G4 + publish-track GPU jobs.

## Acceptance (met)

| Criterion | Evidence |
|---|---|
| Role/dtype page planner | `page_manager.py`, `mixed_kv.plan_int4_mask` |
| True packed BF16 vs INT4 payloads | `packed_mixed_cache.py` · `storage=packed` in `mixed_kv_run` |
| FlashInfer LSE multicall parity | `w6e` / `w6i` / `w6j` **PARITY_PASS** |
| FI decode without HF `materialize_hf_past` | `fi_mixed_decode.py` + `qwen3_fi_shim.py` · `w9` / D4 M3c |
| Quality @ packed path | lock-240 `mg_b` **MG_LOCK240_PASS** |
| Bytes + latency | `mg_a` peak/payload · `d4_latency_m3c` **D4_M3_PASS** |

## Accepted limitation (not a reopen)

**Cold scratch:** INT4 pages are dequantized to a BF16 GPU scratch for FI attend.
Peak CUDA can stay near FullKV; **payload bytes** remain the compression claim
(~0.72× measured / ~0.47× modeled). Documented in `mg_a_peak_mem_gpu5_r1`.

Killing cold scratch (page-streamed INT4 attend with no full-layer BF16 expand) is
**post-core** systems work — not required to close D3 for the DeepMind-track science claim.

## Do not reopen for

- Soft-INT4 PriorityBench quality hunts (falsified)
- LongBench / full RULER matrices
- Expanding D4 latency `n` for optics
