# PriorityBench-A — exact dataset (W3 lock)

**Manifest:** `data/prioritybench/manifests/w3_lock.json`  
**SHA256:** `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89`  
**n = 240** · audit PASS (`docs/audit_w3.md`)  
**Model pin for scoring runs:** Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218` (Gemma secondary is separate / reduced)

This is **not** LongBench/RULER. It is a custom **agent-reliability** bench: long chat
transcripts where the gold answer depends on early structured state that eviction can destroy.

---

## First principles — what each example is doing

Every example is a **multi-turn chat** padded to a target length (~8k / 16k / 32k tokens):

1. **Early turns** plant the thing that matters (tool contract, constraint, or ID).
2. **Long filler** (benign chatter / docs) buries that state in the middle of the KV cache.
3. **Final user ask** requires using the early state correctly.
4. **Scorer** is programmatic (0/1): schema/regex/exact slot — not LLM-as-judge.

**What we measure:** does the model still obey the planted state after long context?
**What we stress with PriorityKV:** if you **evict/compress** the wrong KV pages, that
score collapses even when “average” generation looks fine.

Arms on this bench typically:

| Arm | Meaning |
|---|---|
| FullKV | Keep everything (upper bound) |
| Uniform keep / uniform INT4 | Drop or demote without caring about roles |
| Structure keep / structure-mixed | Protect system/tool/constraint/sink/recent |

---

## The three tasks (80 examples each)

### 1. `tool_schema` (80)

**Job:** Emit a **valid tool call** matching an early JSON/schema-style contract
after long filler.

**Templates (10 each):**  
`search_docs`, `read_file`, `sql_query`, `http_get`, `create_ticket`, `set_config`,
`schedule_job`, `send_email` (all `.v1`).

**Scorer:** JSON / required fields / enums (deterministic schema subset).

**Buried:** 0/80 in the lock (W2d preserve filled quota) — gold is still early; filler is long.

### 2. `instruction_supersession` (80)

**Job:** Follow the **latest** constraint when an earlier one conflicts
(e.g. format or language flip).

**Templates:**  
`format_flip.v2` (30) · `language_flip.v2` (30) · + **buried** variants (10+10).

**Scorer:** Regex / format checkers for the *final* constraint (v2 templates:
**no gold leakage** in the FINAL ask).

**Buried:** 20/80 — gold constraint buried deeper in filler (harder for naive structure tags).

### 3. `multi_turn_state` (80)

**Job:** Reuse an early-turn **ID / path / preference** verbatim later
(order_id, file_path, user_pref).

**Templates:**  
`order_id.v2` (21) · `file_path.v2` (20) · `user_pref.v2` (19) · + buried variants.

**Scorer:** Exact-match slots.

**Buried:** 20/80.

---

## Strata (how the 240 are sliced)

| Axis | Counts |
|---|---|
| Category | 80 / 80 / 80 |
| Context length | 8k: **83** · 16k: **81** · 32k: **76** |
| Split | calibration **92** · validation **49** · test **99** |
| Buried flag | buried **40** · plain **200** |

Splits are assigned by stable hash (`assign_split`). **Do not retune locked test IDs**
after the audit SHA is written.

JSONL split files are **gitignored**; regenerate with:

```bash
PYTHONPATH=src uv run python scripts/mk_bench.py --mode w3_lock
PYTHONPATH=src uv run python scripts/audit_bench.py
```

---

## What runs use this dataset

| Run family | Selection |
|---|---|
| Lock-240 quality (`mg_b`) | All 240 matching · Full / uniform / structure @ int4_frac=0.75 |
| Matched-keep stress | Small calibration slices · keep_frac sweeps |
| Publish Gemma reduced | n≈14 calibration @ 8k only (Gemma 8192 cap) |
| Guardrails publish | Separate RULER/SCBench/MATH probes — **not** this 240 |

Code: `src/prioritybench/` · templates under `src/prioritybench/templates/`.
