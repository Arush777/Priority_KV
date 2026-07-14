# Decisions log (solo)

Append-only. Newest at bottom.

## 2026-07-14 — Dual-machine ops

- **Decided:** Solo ownership (no A/B split). All workstreams owned by Arush.
- **Decided:** Cursor agents develop on CCC/login checkout and push to `Arush777/Priority_KV`.
- **Decided:** H200 is human-operated only: `git pull` + `uv` + GPU runs. No agents on H200.
- **Decided:** Env manager is `uv` with `pyproject.toml` + lockfile; GPU extras via `uv sync --extra gpu`.
- **Decided:** Primary model pin remains `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`.

## 2026-07-14 — Shared H200 etiquette

- **Decided:** Hard cap of **2 GPUs** on the shared 8× H200 host. Default `CUDA_VISIBLE_DEVICES=6,7` (override only if busy).
- **Decided:** Operator-facing scripts are bland (`scripts/sync.sh`, `scripts/check.sh`); no project/model slogans in shell banners.

## 2026-07-14 — Git identity + W1 start

- **Decided:** All commits authored/committed as `Arush777 <153831754+Arush777@users.noreply.github.com>` (never CCC/IBM host identity).
- **Decided:** W1 FullKV compare CLI is `scripts/cmp_gen.py`; results under `$PRIORITYKV_SCRATCH/runs/`.

## 2026-07-14 — W1 byte model freeze (pre)

- **Measured (Qwen3-8B GQA 36L×8H×128d):** all-INT4 realized floor ≈ **29.7%** of FullKV BF16 (payload+scales+zp+page table).
- **Implication:** budget **25% is infeasible** without eviction (matches plan). Budget **30%** leaves almost no BF16 headroom (~0.4% of tokens ≈ 144 toks @ 32K); protected pages must be tiny or we treat 50% as the primary quality operating point and 30% as a stress budget.
- **Geom pin:** `QWEN3_8B_KV = (36, 8, 128)` in `src/prioritykv/byte_model.py`; table in `configs/w1_byte_budget.json`.
- **W1 PriorityBench pilot:** 40 `tool_schema` examples, 8 templates, seeds in `data/prioritybench/manifests/w1_pilot.json`.

## 2026-07-14 — W1 closed / W2 start

- **W1 FP8 smoke:** `cmp_fp8.py` → exact=0.850 tok=0.926 pass=1 (S1 runnable on H200). Defer `prep_fp8.py`/llmcompressor to S1 freeze.
- **W2 start:** page manager + structural tagging + protected invariants (CPU reference). Grow PriorityBench with instruction_supersession templates.

## 2026-07-14 — W2 H200 confirm

- **Page smoke (H200):** `check_pages.py` → seq_len=25282, pages=1581, bf16=274 / int4=25008, within_budget=true, invariants_ok=true.
- **W2 pilot:** `mk_bench.py --mode w2` → n=120 (cal 56 / val 27 / test 37); manifest `data/prioritybench/manifests/w2_pilot.json`.

## 2026-07-14 — W2 quality pilot harness

- **Decided:** First agent-reliability compare is `scripts/run_pilot.py` (15× cal/8k: 10 tool + 5 supersession; FullKV vs FP8; deterministic scorers).

## 2026-07-14 — W2 pilot result + supersession fix

- **8k pilot (r1):** full=0.800 fp8=0.800 delta=0; tool_schema 1.0/1.0; supersession 0.40/0.40. Failures were format_flip scorer (looked for format *names* in prose), not FP8.
- **Fix:** format_flip now requires explicit `[[FMT:...]]` tags (rev 2). Re-run 8k then 16k pilots.
- **SnapKV:** scaffold only (`scripts/snap_status.py`); not runnable yet.

## 2026-07-14 — 8k pilot rev2 + 16k OOM-length fix

- **8k r2:** full=1.000 fp8=1.000 delta=0; both categories 1.0 — tag fix worked; FP8 still no harm at 8k.
- **16k r1 failed:** chat-templated prompts exceeded max_model_len 20480. Bumped to 32768 + defensive trim.

## 2026-07-14 — 16k pilot r2

- **16k r2:** full=0.933 fp8=0.933 delta=0; tool_schema 1.0/1.0; supersession 0.80/0.80.
- **Read:** longer context slightly hurts supersession on FullKV already; FP8 still tracks FullKV (no extra agent damage yet). INT4 / 32k / harder templates are the next stress.

## 2026-07-14 — language_flip case fix

- **16k miss:** `...language_flip...s20271049` scored 0 because output had `Bravo` vs required `bravo` (case-sensitive). Flags now include `IGNORECASE`.

## 2026-07-14 — choose multi_turn next

- **Decided:** Highest-signal next step is `multi_turn_state` (exact ID/path recall), not 32k FP8 (likely still delta≈0) or SnapKV wiring yet. Target ~145 with w2b pilot + 16k 3-category quality run.

## 2026-07-14 — w2b 16k 3-cat pilot

- **Result:** full=1.000 fp8=1.000 delta=0; tool_schema / instruction_supersession / multi_turn_state all 1.0/1.0 at 16k (n=15).
- **Read:** FP8 KV is too gentle to surface PriorityBench failures on these templates at ≤16k. Next stress must be **stronger compression (uniform INT4)** or **32k + harder adversarial templates**, not another FP8 smoke.

## 2026-07-14 — W2c INT4 crash fix

- **Symptom on H200:** FullKV+FP8 OK; INT4 fell through to fake path and crashed with `ValueError: too many values to unpack (expected 2)` — Qwen3/HF cache layers are not plain `(k, v)` pairs.
- **Fix:** mutate DynamicCache / `.layers` / legacy ≥2-tuples safely; prefer `cache_implementation="quantized"`; checkpoint vLLM partial JSON before INT4.

## 2026-07-15 — Stop gentle pilots; run DropKeep ~60× stress

- **W2d still perfect** because quanto never engaged (`group_size` ≠ `q_group_size`) and uniform INT4 fake-quant is too weak for these tasks.
- **Decided:** next decisive H200 job is `scripts/run_stress.py` — FullKV vs StreamingLLM-style **sink+recent DropKeep** (~16+256 keep ≈ **60×** at 16k). Expect multi_turn_state crash. That is the information-loss signal, not another soft INT4 1.0.
- Fixed quanto kw to `q_group_size` for later; not the stress focus.

## 2026-07-15 — DropKeep stress HIT (G2 path a open)

- **Result:** `n=14 full=1.000 drop=0.000 d_drop=-1.000 compression≈63.8x`
- **Cats (full/drop):** supersession 1.00/0.00 · multi_turn 1.00/0.00 · tool_schema 1.00/0.00
- **Read:** first real info-loss on PriorityBench. Soft FP8/INT4 at ≤16k were too weak; ~64× eviction destroys all three agent categories while FullKV stays perfect.
- **Next:** keep-budget sweep (512→4k) for the drop-off curve, then structure-protected recovery at matched bytes.

## 2026-07-15 — Sweep flat zeros: two real bugs / physics

- **Sweep r1:** all recent=256…4096 gave drop=0.0. Compression × changed (so budgets applied) but:
  1. **Physics:** sink+recent never keeps early IDs until `recent ≳ seq_len − id_pos` (~7–15k here). 256–4096 still deletes the hold ID on 8k/16k.
  2. **Implementation bug:** KV-cache slicing without RoPE/position fix → decode garbage at every budget (would also zero a true keep_all if we had included one).
- **Fix:** prompt-level sink+recent concat + normal `generate` (RoPE-safe). Sweep r2 adds `recent=999999` keep_all control (must ≈ FullKV) and larger windows (7k/12k) to see the middle of the curve.

## 2026-07-15 — G1 freeze (W2 close)

Fable (senior RE review) confirmed this freeze with two job-4 corrections (fractional budget + random arm).

**Frozen baselines**
- **S0 FullKV** — vLLM BF16; pilots green.
- **S1 FP8** — delta=0 vs FullKV on PriorityBench ≤16k (w2 / w2b / w2c); cite those runs.
- **Q_dropkeep** — prompt-level sink+recent as **interim eviction baseline** (StreamingLLM-style stand-in). Kill ~64×; RoPE-safe sweep keep_all=1.0 control.

**Deferred (must be written, not silent)**
- **Q2 uniform INT4** — deferred to W3 kernels / working quanto path. **Blocking for G2 path (a).**
- **Q3 SnapKV** — scaffold only; ≤4-day attempt in W3 else keep StreamingLLM/DropKeep substitution per plan §9.
- **Guardrails** — `scripts/run_guardrails.py` stubs SKIPPED this week; **must run for real before W4 G2** (guardrail move <1pt).

**W2-close H200 job (G2 path b pilot):** `scripts/run_stress_structured.py` — FullKV vs {uniform, structure, random, keep_all} at **matched keep_frac=0.25** on the 14-ex stress set; per-length breakdown required.

## 2026-07-15 — Structured keep HIT (G2 path b signal)

- **Run:** `stress_structured_25_r1` · n=14 · keep_frac=0.25 · reuse FullKV from dropkeep kill
- **Results:** full=1.000 · **uniform=0.000** · **structure=1.000** · **random=0.000** · **keep_all=1.000** (gate OK)
- **Cats:** structure 1.00 all three; uniform/random 0.00 all three; both 8k and 16k
- **Read:** at matched 25% keep, structure-protected retention restores agent reliability; sink+recent and random-at-matched-budget do not. This is the PriorityKV / G2 path-(b) pilot signal (≥3pt oracle structure vs uniform — here +100pt).
- **Caveat to scrub next:** structure=1.0 everywhere is very clean — confirm OTHER/short-turn tagging isn't over-protecting; denser keep_frac sweep + page-level (not prompt-gather) path in W3.

## 2026-07-15 — Fable on structured HIT: MIXED

- **Verdict:** LEGIT on-benchmark path-(b) signal; bimodal 1/0 is expected because bench puts gold in short turns and policy protects short turns. Scope is "state length-separable from filler," not all traces.
- **Fix applied:** removed `"FINAL" in content` → RECENT oracle keying.
- **Next before W2 close:** buried-state adversarial H200 (`configs/stress_structured_25_buried.yaml`) — expect structure to drop; if still 1.0, leak.

## 2026-07-15 — Buried-state adversarial (W2 close)

- **Run:** `stress_structured_25_buried_r1` · buried=true · keep_frac=0.25 · n=14
- **Results:** full=1.000 · uniform=0.000 · **structure=0.429** · random=0.000 · keep_all=1.000 (gate OK)
- **Cats (structure):** supersession 0.00 · multi_turn 0.00 · **tool_schema 1.00** · len 16k:0.67 / 8k:0.00
- **Read:** structure **drops** when gold is buried in long filler (no leak). Remaining structure win is mostly TOOL/SYSTEM-tagged schemas (role priors, not length). Scopes W2 claim: structure wins when state is role/length-separable; buried free-form state needs better tagging / page risk in W3.
- **W2 status:** closed on evidence. G1 freeze stands. Next = W3 (INT4 path, SnapKV attempt, denser atlas, page-level protect).

## 2026-07-15 — W3 lock + CPU package (Fable GO)

- **Manifest:** `data/prioritybench/manifests/w3_lock.json`
- **SHA256:** `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89`
- **Audit:** `docs/audit_w3.md` · n=240 · 80/cat · w2d_preserved=145 · buried 20/80 for super+multi; tool 0 (W2d filled quota)
- **CPU landed:** mixed BF16/INT4 dequant-then-attend ref (`mixed_cache_reference.py`); INT4 path tests; page-granularity keep (floor to token budget); `allow_fake_fallback=False` assert mode; SnapKV/quanto loud-skip via `scripts/run_w3_baselines_check.py`
- **Cut (Fable D):** `label_page_perturb` deferred W4; FlashInfer multi-call deferred; attention-KL deferred
- **H200 next:** `configs/w3_structured_paged.yaml` + install quanto/kvpress then `configs/w3_int4_assert.yaml`

## 2026-07-15 — Handoff doc for INT4 / quanto_cuda

- Collaborator continues from **`docs/HANDOFF_W3_INT4.md`** (Opus-reviewed language).
- Page stress already green (`w3_structured_paged_r1`); active bug = `quanto_cuda` JIT under assert-no-fake.

## 2026-07-15 — Implementation plan → v2.1 execution overlay

- Rewrote `docs/IMPLEMENTATION_PLAN.md` with status snapshot, G1 deferrals, D1/W3 done-vs-cut, baseline table as lived, and week overlay (✅/🚧/⏸).
