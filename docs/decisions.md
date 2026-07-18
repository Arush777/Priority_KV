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

## 2026-07-15 — H200 remote job pipeline

- **Decided:** Keep “no coding agents on H200”; automate only pull/run via `scripts/remote_worker.sh` + in-repo `jobs/pending|done|failed` queue.
- **Decided:** Commands must be allowlisted (`python scripts/*.py` or `uv run python scripts/*.py`); single job at a time; still 2-GPU cap.
- **Decided:** Full run JSON stays on scratch (`$PRIORITYKV_SCRATCH/runs/`); agent pulls via `scripts/fetch_results.sh` (rsync) into gitignored `scratch_mirror/`. Optional thin `jobs/status/*.json` push from worker — not a substitute for rsync.
- **Decided:** Idempotency via `$PRIORITYKV_SCRATCH/logs/<id>.status` so a reappearing pending file does not re-run.

## 2026-07-15 — Q2 uniform INT4 assert GREEN on H200

- **Blocker was:** `quanto_cuda` Marlin JIT with default `-std=c++17` fails ATen `List_inl.h` on this nvcc/g++ host; toolkit major matched (nvcc 13.x / torch cu130) — not the bug.
- **Fix:** `prioritykv.cxx20_cuda_ext` forces `-std=c++20` on `torch.utils.cpp_extension.load` in the **same process** as `run_pilot3` / `int4_baseline` (prebuild-alone was insufficient — pilot re-JIT’d without the patch).
- **Evidence (`w3_int4_assert_r4`, exit=0):** `modes=['hf_cache_implementation_quantized']` · n=6 · int4_mean=1.000 · all six rows that mode · **not** `fake_groupwise_prefill` · `allow_fake_fallback: false` unchanged.
- **Out:** `$PRIORITYKV_SCRATCH/runs/w2c_pb_quality/w3_int4_assert_r1.json` · log `…/logs/w3_int4_assert_r4.log`
- **Note:** `full=0/fp8=0` expected under `--modes int4_only` (vLLM arms skipped).

## 2026-07-15 — W3 CLOSED

- **Package:** lock 240 + SHA `fc44b966…` · auto-audit · **15% dual audit PASS** (`docs/audit_w3_dual.md`, n=36) · page structure 0.643 · Q2 real INT4 · assert-no-fake · CPU mixed ref + LSE multicall parity tests.
- **Q3:** `scripts/run_snapkv_attempt.py` — `uv sync --extra kvpress`; if SnapKVPress missing → **LOCK_Q_DROPKEEP** as permanent eviction interim (loud, not silent).
- **Carry to W4 (not W3 blockers):** denser keep_frac structure sweeps · H200 guardrails numbers · formal G2 sentence · FlashInfer CUDA (CPU LSE done).

## 2026-07-15 — W4 start (systems + G2 evidence)

- **Landed:** `label_page_perturb.py` + `configs/linear_risk_fit.json` · `run_guardrails.py` (real local RULER/SCBench-style probes) · `mixed_attend_kv_multicall` LSE merge == dense · `flashinfer_multicall` loud-skip · denser configs `w4_structured_paged_{015,035}.yaml`.
- **SnapKV attempt:** `IMPORT_OK` (`kvpress.SnapKVPress` available) — matched-byte quality pilot still open; DropKeep remains interim until that pilot lands.
- **Guardrails H200 (`guardrails_w4_r2`):** **PASS** · gate tasks `ruler_vt`+`scbench_choice` max|Δ|=**0.0** @ keep_frac=0.50 · NIAH/MT logged as stress diagnostics (expected DropKeep-sensitive).
- **H200 ops note:** set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (remote_worker + stress script) after CUDA-fork EngineCore failures following HF loads.

## 2026-07-15 — G2 FORMALLY CLOSED (path b)

- **Criterion (b):** structure-aware keep beats uniform by ≥3 points at matched budget **and** guardrails move ≤1pt.
- **Evidence:**
  - Page structure @ `keep_frac=0.25`: structure **0.643** vs uniform **0.000** (`w3_structured_paged_r1`) — margin ≫ 3pt.
  - Token structure @ 0.25 and buried-scope already documented in W2/W3 decisions.
  - Guardrails gate max|Δ|=**0.0** on VT/choice (`guardrails_w4_r2`).
- **Not claimed:** path (a) ≥5pt PriorityBench drop from uniform INT4 (still soft @8k n=6 with int4=1.000). Measurement/serving story proceeds on **path (b)**.
- **Follow-ups (non-blocking):** denser keep_frac 0.15/0.35 confirmatory sweeps · SnapKV matched-byte pilot · FlashInfer CUDA wiring.

## 2026-07-15 — denser keep_frac structure sweeps GREEN

- **Runs:** `w4_structured_paged_015_r2` → `w4_structured_paged_015_r1.json` · `w4_structured_paged_035_r5` → `w4_structured_paged_035_r1.json` (page, n=14, Qwen3-8B).
- **Means (uniform / structure / random / keep_all):**
  - `keep_frac=0.15`: **0.000 / 0.643 / 0.071 / 1.000**
  - `keep_frac=0.25` (ref `w3_structured_paged_r1`): **0.000 / 0.643 / 0.286 / 1.000**
  - `keep_frac=0.35`: **0.000 / 0.643 / 0.429 / 1.000**
- **Read:** structure mean flat at 0.643 across denser budgets on this set (still tool=1.0, supersession=1.0, multi_turn=0.375); uniform stays dead; random rises with budget (lottery). Confirms G2 path (b) is not a single-operating-point artifact.
- **Ops:** shared H200 contention → `scripts/wait_gpu_and_run.py` picks any 2 GPUs with ≥95 GiB free.

## 2026-07-15 — W4 missing links closed (docs + pilots)

- **FlashInfer CUDA:** **DEFERRED_W5_W6** — `flashinfer_multicall.status()` loud-skip; CPU `lse_merge_pair` / `mixed_attend_kv_multicall` remain correctness oracle (`tests/test_lse_and_risk.py`).
- **Atlas fold:** `scripts/run_atlas_fold_w4.py` → `docs/atlas_w4_structure_rows.jsonl` (0.15/0.25/0.35 arm means + rows when present); `docs/failure_atlas.md` updated.
- **Page-perturb / linear risk:** already landed (`label_page_perturb.py`, `configs/linear_risk_fit.json`) — seed fit only; not yet wired into `keep_policy` (honest scope).
- **SnapKV matched-byte pilot:** wired `scripts/run_snapkv_quality.py` + `configs/w4_snapkv_matched.yaml` (FullKV vs DropKeep@~4k keep vs SnapKVPress `compression_ratio=0.75`); H200 job `w4_snapkv_quality_r1`. Until results land, DropKeep remains eviction interim; decision auto-written into run JSON (`Q3_PARTIAL` or `LOCK_Q_DROPKEEP`).

## 2026-07-15 — Q3 SnapKV matched-byte → LOCK_Q_DROPKEEP

- **Run:** `w4_snapkv_quality_r1` · n=14 · full=1.000 · dropkeep=0.000 · snapkv generate **failed** (`KeyError: cache_position` under HF generate + SnapKVPress hooks).
- **Decision:** **LOCK_Q_DROPKEEP** — DropKeep remains permanent eviction interim. Import-only SnapKV is not a quality baseline.
- **Out:** `$PRIORITYKV_SCRATCH/runs/snapkv_quality/w4_snapkv_matched_r1.json`

## 2026-07-15 — W5 start: P2 structure_risk wired

- **Landed:** `load_linear_risk_config` · `structure_risk` keep policy (page+token) ranks residual budget by `score_page` · `PageManager.enforce_budget` demotes lowest risk first within a role · config `configs/w5_p2_structure_risk.yaml` · job `w5_p2_structure_risk_r1`.
- **Claim scope:** matched-keep prompt ablation (Q7 `structure` vs P2 `structure_risk` vs uniform) — **not** end-to-end mixed BF16/INT4 serving yet.
- **Next:** H200 P2 pilot numbers · FlashInfer CUDA (still deferred) · G3 allocator ablations.

## 2026-07-15 — P2 structure_risk HIT on H200

- **Run:** `w5_p2_structure_risk_r1` · n=14 · page · keep_frac=0.25
- **Means:** uniform **0.000** · structure (Q7) **0.643** · **structure_risk (P2) 1.000** · keep_all 1.000
- **Category:** P2 lifts multi_turn 0.375→**1.000**; tool/supersession stay 1.0
- **Read:** linear-risk ties are not Q7-equivalent on this set (falsifies “Q7 == P2” worry for this pilot). Still a **prompt-level keep ablation**, not mixed BF16/INT4 serving.
- **Out:** `$PRIORITYKV_SCRATCH/runs/stress_structured/w5_p2_structure_risk_r1.json`

## 2026-07-15 — W5/W6 continue (Q6 FixedHot + FlashInfer probe)

- **Ops hygiene:** Coding agents (Cursor/Claude) **never** on H200 — only laptop/agent box writes code + pushes `jobs/pending`; H200 runs `pkworker` + allowlisted `python scripts/*.py`. Status checks may use SSH as the human collaborator; do not install or launch IDEs/agents on `dgre2`.
- **Q6 FixedHot:** `fixed_hot` / `fixed_hot_pages` policy (prefix-hot after sink+recent) · config `configs/w5_q6_fixedhot.yaml` · job `w5_q6_fixedhot_r1`.
- **W6 FlashInfer:** `scripts/run_flashinfer_probe.py` + `flashinfer_multicall.probe()` · job `w6_flashinfer_probe_r1` (loud SKIP/IMPORT_OK; multicall kernel still not wired).

## 2026-07-15 — Q6 FixedHot pilot + FlashInfer probe results

- **FixedHot (`w5_q6_fixedhot_r1`, n=14, page, keep_frac=0.25):**
  - uniform **0.000** · **fixed_hot 1.000** · structure (Q7) **0.643** · structure_risk (P2) **1.000** · keep_all 1.000
  - **Read:** On this unburied set, **prefix-hot FixedHot ties P2** (both perfect). Role-aware P2 is **not uniquely required** here — gold may sit early + in recent. **Buried adversarial is the discriminator** (next).
- **FlashInfer probe (`w6_flashinfer_probe_r1`):** **IMPORT_OK_CUDA_TOUCH** · flashinfer **0.6.13** on H200 · CUDA touch OK · multicall still unwired.
- **Next jobs:** `w5_p2_buried_r1` (FixedHot vs Q7 vs P2 under bury) · `w6_flashinfer_lse_parity_r1` (tiny FI multi-call LSE vs CPU).

## 2026-07-15 — Buried FixedHot vs P2 + FlashInfer LSE parity

- **Buried (`w5_p2_buried_r1`, n=14, page, keep_frac=0.25, buried=true):**
  - uniform **0.000** · **fixed_hot 1.000** · structure (Q7) **0.429** · **structure_risk (P2) 1.000** · keep_all 1.000
  - Q7 matches W2 buried scope (multi_turn→0). **FixedHot still perfect** — bury did **not** discriminate FixedHot from P2 on this set (gold still reachable via prefix+recent keep). Further discriminator needed (harder bury / mid-only keep / lock test split).
- **FlashInfer LSE parity (`w6_flashinfer_lse_parity_r1`):** **FAILED** `exit=1` — FlashInfer JIT `single_prefill_with_kv_cache` Ninja build failed for head_dim=32 SM90 (package import OK; tiny custom-dim kernel path broken). CPU LSE multicall remains the correctness oracle. Retry with model-native head dims (e.g. 128) or prebuilt kernels later.

## 2026-07-15 — Root-cause of FixedHot≡P2 + mid-context discriminator

- **Diagnosis (setup/eval-design, not a novelty failure):** `pad_with_filler_turns`
  (`templates/base.py`) emits `[system, <short gold turns>, <filler…>, FINAL]`, so
  structure-critical state lives in the **prefix**. `select_fixed_hot_pages` keeps
  sink+recent+**lowest page ids (prefix)** → it grabs the same early gold pages as
  role/risk-aware P2. Buried-in-place only lengthened those prefix turns; it never
  moved gold out of the prefix, so FixedHot could not be separated from P2 (both 1.0).
- **Fix:** `relocate_state_to_middle` (`baselines/buried_state.py`) keeps leading
  system turns + FINAL fixed and re-inserts the gold block at the middle of the
  filler band (gold order preserved → supersession/multi-turn semantics intact).
  Wired via config flag `relocate_middle` in `structured_stress.py` + `--relocate-middle`.
  Tests: `test_relocate_moves_gold_out_of_prefix`, `test_relocate_preserves_gold_order`.
- **Prediction:** with gold at mid-context (config `w5_p2_middle`, keep_frac=0.25),
  uniform ~0 and **fixed_hot should collapse** (prefix is now filler), while
  `structure`/`structure_risk` still retrieve state by role → clean FixedHot ≠ P2.
  Tool-schema gold sits in the system message (stays prefix/SYSTEM-protected) so tool
  ties are expected; discriminator lives in supersession + multi_turn. Job `w5_p2_middle_r1`.

## 2026-07-15 — Mid-context discriminator RESULT (FixedHot separated)

- **`w5_p2_middle_r1` (n=16, page, keep_frac=0.25, relocate_middle=0.5, exit=0):**
  - uniform **0.000** · **fixed_hot 0.125** · structure (Q7) **0.688** · structure_risk (P2) **0.688** · keep_all 1.000
  - By category: fixed_hot super **0.00** / multi **0.00** / tool **1.00**; structure & P2 super **1.00** / multi **0.38** / tool **1.00**.
- **Read (primary goal MET):** with gold at mid-context, **FixedHot collapses 1.000→0.125** while structure/P2 hold at 0.688 — a clean position-heuristic vs structure separation. FixedHot's residual is entirely tool_schema (system-message gold, stays prefix/SYSTEM-protected, as predicted). Novelty claim is now demonstrated honestly: *structure-aware keep ≫ uniform AND prefix (FixedHot) when critical state is not at the edges.*
- **Honest nuance:** P2 **==** Q7 here (both 0.688). The earlier P2>Q7 gap was partly a prefix artifact; with position controlled the linear-risk tie-break adds nothing on this set, and both leave `multi_turn` at 0.38. Defensible claim = "structure-aware ≫ position heuristics," not "P2 ≫ Q7".
- **Next:** (1) close G3 on the FixedHot≠structure discriminator; (2) chase `multi_turn` 0.38 (why role-aware keep still misses it — likely OTHER-role short-turn budget contention); (3) a P2≠Q7 test only if the linear-risk refinement is to be claimed separately.

## 2026-07-15 — Systems half started: real mixed BF16/INT4 KV forward

- **Why:** all prior wins were prompt-level *keep* (drop+regenerate at full BF16) — the
  headline "mixed BF16/INT4 paged serving" was never exercised on a real forward. This
  is the biggest goal-alignment gap.
- **Approach (Stage 1–2, avoids blocked FlashInfer JIT + risky custom Cache subclass):**
  reuse the green uniform-INT4 fake-quant path (`_fake_quant_past`) but make it
  **per-position**. `mixed_kv.plan_int4_mask` picks which token positions store INT4 at a
  matched byte budget (`int4_frac`): `structure` protects sink+recent+roles and demotes
  lowest-risk first; `uniform` demotes role-blind evenly (same INT4 count → byte-fair).
  `mixed_kv_run.run_transformers_mixed_kv` prefills full, INT4-round-trips only the masked
  prompt-KV positions in-place (groupwise, same error model as Q2), then greedy-decodes.
- **Honest scope:** this measures the *quality frontier* of role-aware mixed precision at a
  matched byte budget — NOT yet wall-clock memory/latency (true packed cache + FlashInfer
  is a later stage). Realized INT4 fraction is logged per example.
- **Tests:** `tests/test_mixed_kv.py` (matched budget, sink/recent never INT4, structure
  protects roles). Full suite 69 passed / 3 skipped.
- **Job:** `w6_mixed_serve_r1` — FullKV vs uniform-INT4 vs structure-mixed @ int4_frac=0.75,
  mid-context set. **Desired result: structure ≫ uniform at equal bytes.**

## 2026-07-15 — Mixed-serve r1 RESULT (ops unblocked; quality claim soft)

- **Job status:** `w6_mixed_serve_r1` **done, exit=0** (finished 2026-07-15T14:40:53Z). Queue idle.
- **Ops blocker (git divergence):** **SOLVED** — worker status commit was rebased onto origin; ff-only pulls resumed; job claimed and ran.
- **Systems plumbing:** **WORKS** — real forward + per-position INT4 round-trip executed; uniform/structure **matched** at int4_frac_realized **0.75** (byte-fair by construction).
- **Quality claim at int4_frac=0.75:** **NOT yet shown.** Means: full **1.000** · uniform **1.000** · structure **1.000** (all cats 1.0). Soft INT4 at 75% does not hurt this mid-context set — same lesson as G2 path (a) / Q2 soft @8k. Discriminator must be a **harder** budget (higher int4_frac, e.g. 0.90–0.95, or true drop+INT4) before structure vs uniform can separate.
- **Read:** first systems half is on the board (artifact exists and runs on H200). Remaining gap for the *claim* is budget severity, not plumbing. Next: retry `w6_mixed_serve` at higher `int4_frac` (or INT4+evict) until uniform drops while structure holds.

## 2026-07-15 — Tackle soft-INT4 + FlashInfer head_dim blockers

- **FlashInfer:** r1 failed because SM90 `static_assert(HEAD_DIM ∈ {64,128,256})` — head_dim=32 is illegal. Script now defaults to **128**, rejects illegal dims (exit 2), artifact tag `r2`. Job `w6a_flashinfer_lse_parity_r2`.
- **Soft INT4 at 0.75:** three follow-ups enqueued (wiring first, then severity):
  1. `w6b_mixed_serve_zero_r1` — `degrade=zero` (INT0) @ matched 0.75: proves mask/planner (expect structure ≫ uniform).
  2. `w6c_mixed_serve_nbits2_r1` — nbits=2 @ 0.75: harsher groupwise error.
  3. `w6d_mixed_serve_hard_r1` — nbits=4 @ int4_frac=0.92.
- **Code:** `mixed_kv_run` supports `degrade: int4|zero`; configs under `configs/w6_mixed_serve_{hard,nbits2,zero}.yaml`.

## 2026-07-15 — W6 blocker root causes after hard runs

- **Hard-run negatives (all exit=0):** nbits=2 @0.75 and INT4 @0.92 both stayed
  full=uniform=structure **1.000**. Zero @0.75 separated structure **0.688** vs
  uniform **0.312**, proving the role mask changes the real cache.
- **Mixed harness correctness bug:** it degraded `past_key_values` after a full
  prompt prefill, but selected the first generated token from the original,
  undegraded prefill logits. Since exact-format tasks are strongly first-token
  driven, this could hide quantization damage. Fix: split prefill at `n-1`,
  degrade that cache, replay the final prompt token, and derive the first output
  token from the degraded cache. New metadata asserts
  `first_token_from_degraded_cache=true`.
- **FlashInfer merge bug:** r2 compiled and ran at native head_dim=128, but we
  fed FlashInfer LSE into the NumPy natural-log merge. FlashInfer 0.6.13
  historically uses a base-2 LSE contract. Fix: merge CUDA states with the
  library-native `flashinfer.merge_state`; keep NumPy `lse_merge_pair` only for
  the natural-log CPU oracle.
- **Research read:** do not tune thresholds or manufacture a win. Re-run the
  corrected harness. If uniform INT4 remains perfect at the target byte budget,
  the quality-advantage half is falsified for this set and the systems claim must
  pivot to throughput/lower budgets rather than implying a reliability drop.

## 2026-07-17 — Stage-1a FI decode state (council GO)

- **Council:** Fable + Codex Sol 5.6 xhigh → FI decode before D4; refuse silent materialize.
- **Landed:** `src/prioritykv/fi_mixed_decode.py` — `FiMixedDecodeState`, `build_from_packed_cache`, FI chunk attend (≤2 chunks), parity vs dense without `materialize_hf_past`.
- **H200:** `w7_fi_decode_smoke_r1` → **PARITY_PASS** (exit=0).

## 2026-07-18 — Agent-remote control (no H200 laptop)

- **Worker:** `remote_worker.sh` now pushes `jobs/results/<id>/` (summary.json from `out=`, log_tail, nvidia-smi before/after) + richer `jobs/status/*.json` (`decision`/`pass`).
- **Agent helpers:** `./scripts/pull_job.sh [--watch] <id>`, `scripts/diag_nvidia_smi.py`.
- **One-time H200:** `bash scripts/h200_bootstrap_pkworker.sh` (reset --hard + restart tmux pkworker).
- **w8:** confirmed GREEDY_PASS; summary backfilled under `jobs/results/w8_fi_greedy_smoke_r1/`.
- **w8b GPU5:** GREEDY_PASS in 8.8s on single GPU 5.
- **w9:** `mixed_kv_run` flashinfer path uses FI shim decode (no materialize); job `w9_mixed_fi_decode_gpu5_r1` on GPU 5 (full/uniform/structure @ 8k n=3).
- **D4:** `d4_latency_micro_gpu5_r1` enqueued (TTFT/TPOT microbench on GPU 5).

## 2026-07-18 — D4 M1 TTFT harness (Fable GO)

- **Falsified claim from raw D4:** mixing first-call cold dequant / FI JIT into
  `ttft_ms` (structure ~3.5s, uniform ~8s vs FullKV ~43ms). TPOT already better
  on FI (~28 vs ~44ms). Do **not** publish those TTFT numbers.
- **M1 fix:** `eager_prepare_decode` + untimed FullKV/FI warmup; report
  `prefill_ms` / `pack_ms` / `cold_scratch_ms` / `decode_ttft_ms` / `e2e_ttft_ms`.
  Gate: mean `decode_ttft(structure) ≤ 3× decode_ttft(fullkv)` → `D4_M1_PASS`.
- **Job:** `d4_latency_m1_gpu5_r1` on GPU 5. M3 big matrix NO-GO until M1 passes.

## 2026-07-18 — D4 M1 PASS (warmup-corrected decode TTFT)

- **Job** `d4_latency_m1_gpu5_r1`: **D4_M1_PASS** (exit=0, ~118s).
- **decode_ttft:** FullKV **47.7ms** · structure **36.0ms** · uniform **41.6ms**
  (structure/FullKV ratio **0.75** ≪ 3× gate). Prior 3.5–8s figures were
  measurement artifacts (lazy cold + first FI call).
- **TPOT:** FullKV ~49ms · FI ~32ms (still ahead).
- **e2e still broken:** structure e2e ~13.3s = pack ~9.0s + cold_scratch ~3.9s
  + decode; uniform worse (pack ~14.3s + cold ~7.2s). Sol/Fable M2 next:
  GPU pack + kill full BF16 cold scratch before any M3 matrix / bytes claim.

## 2026-07-18 — D4 M2 GPU pack/dequant (auto-loop)

- **Council:** Fable+Sol — attack pack+cold before M3; 1×GPU; torch groupwise
  quant/dequant on device (no custom CUDA). Gate: pack≤200ms, cold≤100ms,
  e2e≤1.15× FullKV.
- **Code:** `int4_path` torch ops; `build_from_hf_prefill(prefer_gpu)`; FI cold
  scratch dequants on device.
- **Job:** `d4_latency_m2_gpu5_r1` on free GPU 5.

## 2026-07-18 — D4 M2 FAIL → M2b batched pack

- **M2** `d4_latency_m2_gpu5_r1`: **D4_M2_E2E_GATE_FAIL**. Structure pack
  4.8s / cold 1.7s (better than 9s/3.9s but far from gate). Uniform worse
  (pack 27s) — per-page launch tax.
- **Council:** batch gather+quantize once/layer; keep pack≤200 cold≤100;
  relax e2e to ≤1.5× FullKV for M2b; don't ship uniform gate.
## 2026-07-18 — D4 M2b PASS (batched pack)

- **Job** `d4_latency_m2b_gpu5_r1`: **D4_M2_PASS** (~12s wall).
- Structure: pack **32ms**, cold **13ms**, decode_ttft **28ms**, e2e **445ms**
  vs FullKV e2e **437ms** (**1.016×**). TPOT **27 vs 44ms**. Scores 1.0.
- Uniform similarly fixed (was 27s pack). Per-page loop was the failure mode.
- **Next:** M3 larger latency matrix (council), then G4 freeze.

## 2026-07-18 — D4 M3 matrix enqueued

- **Council:** Fable GO — ctx {8k,16k}, n=9/ctx, repeats=3 median, 128 tok,
  1×GPU; no 32k/vLLM this job. Sol also GO (wants FP8/32k later).
- **Job:** `d4_latency_m3_gpu5_r1` — `--m3-gate` per-ctx pack/cold/e2e/TPOT.

## 2026-07-18 — D4 M3 FAIL → M3b gate fix

- **M3** `D4_M3_GATE_FAIL`: pack/cold/e2e OK; FI TPOT ~28ms stable; FullKV
  steady ~23ms (M2b's 44ms was cold-start noise). Score 0.89@16k hits FullKV
  too (bench artifact). Early EOS (~12 tok) undercut fixed-length TPOT.
- **M3b:** no EOS stop in latency harness; score relative to FullKV;
  `--tpot-gate-mult 1.25` / `--ttft-gate-mult 1.25`; job on free GPU 5
  (not GPU 1 — COSMOS resident). Shim ≤1.0 TPOT is a later FI-plan milestone.

## 2026-07-18 — D4 M3b dual FAIL → M3c score prefix

- **Dual** `d4_latency_m3b_gpu56_r1`: latency OK (pack/cold/e2e/tpot gates);
  16k PASS; 8k FAIL only `score_ok` (FullKV 1.0 vs FI 0.89 on one http_get —
  post-EOS `</think>` trash after correct JSON polluted scoring).
- **M3c:** score on EOS/`</think>` prefix only; re-run dual GPUs 5+6.

## 2026-07-18 — D4 M3c PASS (dual GPU matrix)

- **Job** `d4_latency_m3c_gpu56_r1`: **D4_M3_PASS** (~6 min wall, GPUs 5∥6).
- **8k structure:** pack 35 ms · cold 14 ms · e2e **1.12×** · tpot **1.20×** · score 1.0
- **16k structure:** pack 48 ms · cold 20 ms · e2e **1.11×** · tpot **1.21×** ·
  score 0.89 (= FullKV; bench artifact, not compression).
- Honest systems claim ready for G4 freeze: e2e≈parity, pack/cold cheap,
  TPOT ~1.2× FullKV (FI shim overhead), quality matched to FullKV.

## 2026-07-19 — Middle-ground close (not LongBench)

- **Scope (locked):** finish original PriorityKV claim on Qwen3-8B only.
  1. PriorityBench lock-240 quality: FullKV / uniform / structure @ int4_frac=0.75
     (packed FI shim) — job `mg_b_lock240_quality_gpu01_r1` **MG_LOCK240_PASS**
     (~23 min wall, GPUs 0∥1). n=240 · full **0.888** · uniform **0.879**
     (Δ−0.008) · structure **0.883** (Δ−0.004) · int4_real=0.75.
     By ctx: 8k/16k all **1.000**; 32k full 0.645 / uniform 0.618 / structure 0.632
     (structure slightly above uniform; both near FullKV — no soft-INT4 quality gap).
  2. Measured peak CUDA mem + packed payload bytes — job `mg_a_peak_mem_gpu5_r1`
     (**DONE** `MG_PEAK_MEM_PASS`: structure peak ~20.5 GB vs FullKV ~23.6 GB;
     measured payload ~0.72×; modeled ~0.47×).
  3. Optional thin guardrails later; **do not** rescale D4 latency n; **do not**
     full LongBench/RULER unless publishing.
- **Code:** `configs/mg_*.yaml`, `scripts/run_mg_peak_mem.py`,
  `scripts/run_mg_lock240_dual.py` (max 2 GPUs), `selection.all_matching`.

## 2026-07-19 — Publish-track reopen (GPU jobs overnight)

- **Scope reopen:** arXiv-first publish plan after middle-ground G4.
- **H200 scratch results (git push had failed; worker then ff-only stuck):**
  1. `pub_a_d4_fp8_compare_gpu01_r1` → **D4_FP8_COMPARE_PASS** (exit=0, GPUs 0,1)
  2. `pub_b_guardrails_gpu5_r1` → **GUARDRAILS_PUB_PASS** (exit=0; ran stale GPU-5 YAML before cancel landed)
  3. `pub_c_gemma_reduced_gpu5_r1` → **SKIP_NO_GEMMA** (exit=0; acceptable)
- **Ops:** H200 `git reset --hard origin/main` + restart `pkworker`. Cancelled redundant pending `pub_b/c_*gpu01*` so they do not re-run; leave `pub_a` pending so worker SKIP-archives and retries status push.
- Claim discipline unchanged: eviction reliability + bytes/latency; soft-INT4 gap stays falsified.

## 2026-07-19 — G4 FREEZE (middle-ground close)

- **Status:** **G4 CLOSED** for the middle-ground definition of done.
- **Manifest:** [`FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml)
  (`freeze_id: G4_MIDDLE_GROUND_2026_07_19`).
- **Canonical jobs pinned:**
  - Reliability (family A): prior matched-keep / zero-degrade artifacts
  - Latency: `d4_latency_m3c_gpu56_r1` (`D4_M3_PASS`)
  - Peak/payload: `mg_a_peak_mem_gpu5_r1` (`MG_PEAK_MEM_PASS`)
  - Lock-240 quality: `mg_b_lock240_quality_gpu01_r1` (`MG_LOCK240_PASS`)
- **Deferred (explicit, not blockers):** thin guardrail re-run; LongBench/RULER;
  paper/blog/outreach/Gemma; FP8 latency bake-off.
- **Read:** project point = agent-KV *reliability under eviction* + *honest packed
  serving metrics* — not a soft-INT4 accuracy win on PriorityBench.

- **FlashInfer r3 (`w6e_flashinfer_lse_parity_r3`, exit=0):**

  `flashinfer.merge_state` gives multicall vs FI-dense max abs **0.000488**
  (vs CPU dense **0.000470**); CPU multicall oracle error `1.35e-9`.
  **PARITY_PASS.** The prior 0.085 error was entirely the LSE contract mismatch.
- **Split-prefill verification:** every row in the replay artifacts records
  `first_token_from_degraded_cache=true`; FullKV remains **1.000**, so replay
  itself does not alter the reference.
- **Corrected mixed results (n=16, int4_frac=0.75, matched masks):**
  - INT4: uniform **1.000** · structure **1.000**
  - 2-bit: uniform **1.000** · structure **1.000**
  - zero stress: uniform **0.3125** · structure **0.6875**
- **Decision:** planner wiring and the structure signal are real, but a
  structure-over-uniform *quantization-quality* win is **not supported** at this
  operating point—even at 2-bit. Stop escalating fake-quant severity: it would
  be result-seeking. Keep the defensible reliability claim (structure beats
  uniform eviction / missing-state stress), and move the mixed-dtype systems
  claim to **packed bytes + TTFT/TPOT/throughput**. A real packed backend, not
  another fake-quant quality sweep, is the next blocker.
