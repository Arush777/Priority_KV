# PriorityKV: Prajna BFCL External-Evaluation Handoff

**Purpose:** give a new Cursor agent enough context to prepare and run the
publication-facing external evaluation on Prajna without relying on the old H200
worker or an interactive GPU shell.

**Important:** Prajna is a **Slurm cluster**. GPU work must run through `sbatch`;
use `squeue`/`sacct` to inspect it. Never run CUDA workloads on the login node.

## 0. Read this first

PriorityKV already has a frozen synthetic-science core and a paper:

- [`../README.md`](../README.md): concise claim and current headline results.
- [`../RESULTS.md`](../RESULTS.md): canonical metrics.
- [`EVIDENCE.md`](EVIDENCE.md): claim registry and negative results.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): environment and frozen artifacts.
- [`../FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml): frozen IDs.
- [`../paper/prioritykv_manuscript.md`](../paper/prioritykv_manuscript.md): paper.

Do **not** modify frozen configurations or overwrite old job IDs. The Prajna
evaluation is a new post-freeze namespace, suggested ID:

```text
EXTERNAL_BFCL_PRAJNA_V1
```

## 1. What the new evaluation must establish

The current strongest result is on PriorityBench-A, a synthetic benchmark built
in this repository:

- Qwen, 25% keep, `n=120`: structure `0.933` vs uniform/random `~0.008`.
- Structure `0.933` vs SnapKV/Pyramid/hybrid `0.900`.
- The four-example structure edge over SnapKV is not significant (`p=0.125`).
- At 5% keep on Llama, SnapKV beats structure on two tested slices.
- INT4 placement does not produce a meaningful quality separation.
- The packed FlashInfer path saves payload memory but currently increases latency.

The missing publication evidence is an evaluation on a benchmark we did not
construct ourselves. The resource-constrained design is:

1. **BFCL V3 multi-turn:** GPU behavioral evaluation.
2. **Public τ-bench trajectories:** CPU-only gold-span retention audit.
3. No τ-bench user simulator and no end-to-end τ-bench rollout.

The target claim is deliberately bounded:

> On externally authored BFCL multi-turn decision points, structure-aware
> retention is compared with blind eviction and real attention-based selectors
> at a matched token budget. A separate generation-free audit on public
> τ-bench trajectories measures whether naturally occurring schemas,
> identifiers, tool results, and conversational constraints survive each
> retention policy.

The τ-bench audit is **mechanistic evidence**, not behavioral task-success
evidence. Never describe it as an end-to-end τ-bench evaluation.

## 2. Fixed resource envelope

- Storage: at most **500 GB** persistent.
- Compute: at most **72 total H100 GPU-hours**.
- Maximum concurrent GPUs: **2**.
- Expected availability: interruptible/resumable blocks, roughly 7–8 hours.
- Recommended execution unit: one H100 per Slurm array task, array concurrency
  capped at `%2`.

If “72 hours” later turns out to mean 72 wall-clock hours on each of two GPUs
(144 GPU-hours), keep the primary design unchanged and use the surplus only for
predeclared extensions.

### Storage budget

| Item | Budget |
|---|---:|
| Qwen3-8B model + HF cache | 40 GB |
| Optional Llama-3.1-8B transfer | 30 GB |
| UV, wheels, CUDA/FlashInfer/JIT caches | 80 GB |
| BFCL + τ-bench data | 20 GB |
| Results, logs, checkpoints | 60 GB |
| Temporary/local staging | 70 GB |
| Reserve | 200 GB |

Never persist full KV tensors per example. Save token IDs, compact keep masks,
outputs, scores, timings, and metadata. Thousands of full 16k/32k KV caches
would exceed the quota.

## 3. Models and context policy

### Primary

```text
Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218
local directory name: Qwen3-8B
thinking: disabled (existing project convention)
```

Use the pinned revision already in `FINAL_RUN_MANIFEST.yaml`.

### Context handling

- Render every BFCL prefix with the pinned Qwen tokenizer before GPU work.
- Record the exact token length.
- Allow room for generation; use a conservative prompt ceiling below the
  configured model maximum.
- Never silently head-tail truncate an external example.
- Mark over-limit tasks as excluded with reason `MODEL_CONTEXT_LIMIT`.
- Report the number and category distribution of excluded tasks.

### Optional transfer

Use the already-pinned Llama-3.1-8B-Instruct only if:

1. the Qwen primary table is complete,
2. paired completeness checks pass, and
3. at least 4 GPU-hours remain.

Do not download or prepare Llama during initial provisioning unless the model
cache can be staged without consuming GPU allocation time.

## 4. Dataset plan

### BFCL V3 multi-turn

The release has 1,000 multi-turn cases:

- 200 base;
- 200 missing parameters;
- 200 missing functions;
- 200 long-context;
- 200 composite.

Pin the exact Hugging Face dataset revision or upstream Git commit. Do not use a
moving `main` revision in paper artifacts.

Primary target: **600 independent conversations**:

| Category | Target |
|---|---:|
| Base | 100 |
| Missing parameters | 100 |
| Missing functions | 100 |
| Long-context | 150 |
| Composite | 150 |

Minimum acceptable table: 400 conversations with preserved category balance.
Stretch: all 1,000 only if the measured pilot throughput proves it fits.

Use **one predeclared decision point per conversation** for the primary paired
statistics. Additional turns may be saved for descriptive analysis but must not
be counted as independent samples.

### τ-bench trajectory audit

Preferred source:

```text
AgentSuite/tau-bench-trajectories
```

Pin its revision. This dataset is small enough to audit on CPU. No user
simulator, tool backend, or generation is required.

Audit all usable trajectories if extraction quality is stable; otherwise freeze
a stratified sample of at least 1,000 trajectories. Cluster summaries by task
and source model so repeated trajectories do not masquerade as independent
behavioral evidence.

## 5. Arms and budgets

Primary BFCL arms at `keep_frac=0.25`:

1. `full` — FullKV control.
2. `structure` — application-visible structure policy.
3. `uniform` — position-blind matched keep.
4. `snapkv` — real attention-based SnapKV implementation.
5. `random` — deterministic seeded matched keep.

Optional:

6. `pyramid` — only after the five-arm table is complete.

Stress subset:

- `keep_frac=0.10`;
- 200 long-context/composite conversations;
- prioritize `full`, `structure`, `uniform`, and `snapkv`.

All non-FullKV arms must keep the same number of tokens for a decision point.
Record requested and realized keep counts. A “SnapKV” result is invalid if the
code silently falls back to DropKeep or another heuristic.

## 6. What is not yet implemented

As of this handoff, the repository has no BFCL or τ-bench integration. Before
submitting the main GPU run, the Prajna agent must add:

```text
scripts/prepare_bfcl_external.py
scripts/run_bfcl_external.py
scripts/score_bfcl_external.py
scripts/audit_tau_retention.py
configs/external_bfcl_prajna_v1.yaml
cluster/prajna/README.md
cluster/prajna/config.example.env
cluster/prajna/bootstrap_cpu.sbatch
cluster/prajna/smoke_h100.sbatch
cluster/prajna/bfcl_array.sbatch
cluster/prajna/tau_audit_cpu.sbatch
```

Names may change slightly, but the responsibilities and output schema below are
required.

Do not reserve the full GPU budget while these components are being debugged.
All parsing, manifest creation, scoring, checkpoint/resume logic, and mocked
policy tests must pass on CPU first.

## 7. Required output and checkpoint schema

Freeze a work manifest in JSONL. Each row is one task-arm work unit:

```json
{
  "work_id": "sha256-stable-id",
  "freeze_id": "EXTERNAL_BFCL_PRAJNA_V1",
  "dataset_revision": "<commit>",
  "task_id": "<bfcl-id>",
  "category": "long_context",
  "decision_turn": 3,
  "model_id": "Qwen/Qwen3-8B",
  "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "arm": "structure",
  "keep_frac": 0.25,
  "seed": 0,
  "prompt_token_count": 14372
}
```

Stable identity must derive from:

```text
dataset revision / task ID / decision turn / model revision /
arm / keep fraction / seed / harness revision
```

Every completed work unit writes a separate result:

```text
$PRAJNA_ROOT/results/external_bfcl_prajna_v1/
  manifest/
    tasks.jsonl
    work_items.jsonl
    exclusions.jsonl
    hashes.json
  points/
    <work_id>.json
  failures/
    <work_id>.json
  shard_logs/
  summaries/
```

Atomic completion protocol:

1. Write `<work_id>.json.tmp`.
2. Flush and `fsync`.
3. Validate required fields.
4. Atomically rename to `<work_id>.json`.
5. On restart, skip only valid completed JSON files.
6. Preserve failure records; never silently drop them.

Each result must contain:

- work ID and all manifest identity fields;
- Git commit;
- `uv.lock` hash;
- model and dataset revisions;
- Slurm job ID, array task ID, hostname, and GPU name;
- prompt hash and prompt token count;
- requested/realized keep count;
- compact retained indices or bit mask;
- raw generated text and parsed tool call;
- official BFCL score components;
- prefill, selection, decode, and end-to-end timings;
- peak allocated/reserved CUDA memory;
- terminal state: success, model failure, scorer failure, OOM, timeout, etc.

### Sharding

Do not launch one Slurm job per decision point because model loading would
dominate. Create shards containing approximately 25–50 work units of the same
arm/model/budget. Each array task:

1. loads the model once;
2. iterates its shard;
3. checkpoints every decision point;
4. skips valid completed points after restart.

Cap concurrent array tasks at two:

```bash
sbatch --array=0-$((N_SHARDS-1))%2 ...
```

## 8. Prajna/Slurm discovery — do not guess cluster settings

The repository does not know Prajna's account, partitions, QOS, module names,
GPU request syntax, filesystem paths, internet policy, or maximum job duration.
Discover them before writing final `.sbatch` directives:

```bash
sinfo
sinfo -o "%P %a %l %D %G"
scontrol show partition
sacctmgr show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null || true
module avail 2>&1 | less
```

Obtain from local Prajna documentation or admins:

- account/project string;
- H100 partition;
- CPU partition;
- QOS;
- GPU request syntax (`--gres=gpu:h100:1`, `--gpus=1`, or site-specific);
- maximum wall time;
- persistent 500 GB filesystem path;
- whether compute nodes have outbound internet;
- whether login-node downloads are permitted;
- CUDA driver/toolkit module.

Record the answers in an untracked file:

```text
cluster/prajna/config.env
```

Track only `config.example.env`; never commit tokens or personal paths.

## 9. Recommended Slurm architecture

### CPU bootstrap job

Purpose:

- clone/update the repository;
- install from `uv.lock`;
- download model and datasets if network is available;
- generate and hash the BFCL manifest;
- run CPU tests.

This job requests no GPU. If outbound network is unavailable on compute nodes,
download on the permitted transfer/login node or stage archives from another
machine. Do not discover this after starting the 72-hour GPU allocation.

### H100 smoke job

Request:

- one H100;
- 8–16 CPU cores;
- 64–128 GB host RAM;
- 1–2 hours;
- no array.

It must verify:

1. `torch.cuda.is_available()`;
2. H100 compute capability 9.0;
3. Transformers model load from the local model directory;
4. FlashInfer import and packed parity;
5. one BFCL FullKV point;
6. one structure point;
7. one real SnapKV point;
8. official scoring;
9. forced interruption and successful resume.

Do not submit the primary array unless all nine pass.

### Primary BFCL array

Request per task:

- one H100;
- one shard;
- at most 7.5 hours;
- array concurrency `%2`.

The 8B model does not need tensor parallelism. Two independent one-H100 workers
are more useful than one two-H100 model.

### CPU τ-bench audit

Request:

- CPU only;
- enough host memory for tokenizer and trajectory parsing;
- array only if necessary;
- no GPU accounting.

## 10. Environment installation on a new Prajna filesystem

Use a persistent root such as:

```bash
export PRAJNA_ROOT=/path/to/500GB/persistent/prioritykv
mkdir -p "$PRAJNA_ROOT"/{repo,scratch,models,datasets,results,logs}
mkdir -p "$PRAJNA_ROOT"/{uv-cache,hf-cache,torch-extensions,xdg-cache}
```

Clone:

```bash
git clone git@github.com:Arush777/Priority_KV.git "$PRAJNA_ROOT/repo"
cd "$PRAJNA_ROOT/repo"
git status
git rev-parse HEAD
```

Set cache paths:

```bash
export UV_CACHE_DIR="$PRAJNA_ROOT/uv-cache"
export HF_HOME="$PRAJNA_ROOT/hf-cache"
export TORCH_EXTENSIONS_DIR="$PRAJNA_ROOT/torch-extensions"
export XDG_CACHE_HOME="$PRAJNA_ROOT/xdg-cache"
export PRIORITYKV_SCRATCH="$PRAJNA_ROOT/scratch"
export PYTHONPATH="$PRAJNA_ROOT/repo/src"
export TORCH_CUDA_ARCH_LIST=9.0
```

Install `uv` if absent, then reproduce the lock:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra gpu --extra kvpress --extra dev
```

Do not use ad-hoc `pip install`. If the lock does not resolve on Prajna, record
the exact error and fix the lock in a separate commit instead of mutating the
environment invisibly.

Create an untracked `.env` with `chmod 600`:

```dotenv
REPO_ROOT=/path/to/prioritykv/repo
PRIORITYKV_SCRATCH=/path/to/prioritykv/scratch
HF_HOME=/path/to/prioritykv/hf-cache
HF_TOKEN=<not committed>
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=<site CUDA path>
TORCH_CUDA_ARCH_LIST=9.0
```

Do not print `HF_TOKEN` into Slurm logs.

### Model download

Download the pinned snapshot into:

```text
$PRAJNA_ROOT/scratch/models/Qwen3-8B
```

Use `huggingface_hub.snapshot_download` with the exact revision. Confirm that
the download is complete before GPU allocation. The existing resolver will use
`$PRIORITYKV_SCRATCH/models/Qwen3-8B` automatically.

### Environment verification inside an H100 allocation

The smoke job should log:

```python
import torch, transformers, vllm, flashinfer
print(torch.__version__)
print(transformers.__version__)
print(vllm.__version__)
print(flashinfer.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```

Then run:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/run_flashinfer_packed_parity.py \
  --head-dim 128 --seq-len 256 --out-tag prajna_h100_smoke
```

Expected FlashInfer result: `PARITY_PASS`. Exact H200 timings need not match.

## 11. GPU-hour schedule

Conservative total:

| Stage | GPU-hours |
|---|---:|
| H100 environment/model/JIT smoke | 6 |
| 20-task five-arm calibration pilot | 8 |
| Primary 400–600 task table | 42 |
| 200-task hard-budget subset | 8 |
| retries/incomplete pairs | 8 |
| **Total** | **72** |

Do not assume the primary table fits until the pilot measures seconds per
task-arm. After the pilot:

```text
available task-arms =
  remaining GPU-seconds / measured median-or-p75 seconds per task-arm
```

Use p75 rather than only the mean because long-context cases dominate runtime.

Decision rule:

- If five-arm 600-task completion fits with 20% reserve, run 600.
- Otherwise run 400 with category balance.
- Do not remove hard examples merely to raise scores or throughput.
- Do not run PyramidKV until the paired five-arm table is complete.

## 12. Monitoring and recovery

Submission:

```bash
JOB_ID=$(sbatch --parsable <site options> cluster/prajna/bfcl_array.sbatch)
echo "$JOB_ID"
```

Monitoring:

```bash
squeue -j "$JOB_ID"
squeue -u "$USER"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

Inspect logs without polling too aggressively:

```bash
tail -n 100 "$PRAJNA_ROOT/logs/<job>_<array-index>.out"
```

On timeout/preemption:

1. do not delete partial output;
2. inspect `sacct` terminal state;
3. rerun only incomplete shards;
4. the runner must skip valid point checkpoints;
5. maintain a failure ledger.

If a job is clearly wrong, cancel it:

```bash
scancel "$JOB_ID"
```

### Signal handling

The `.sbatch` scripts should request an early termination signal if Prajna
supports it:

```bash
#SBATCH --signal=B:TERM@120
```

The runner should handle `SIGTERM`, finish/flush the current atomic checkpoint,
write a shard status file, and exit. Do not depend solely on a shell `trap`;
Python owns point-level result integrity.

## 13. Statistical and reporting requirements

Primary unit: conversation/task.

Report:

- official BFCL overall score;
- schema validity;
- function-name correctness;
- argument correctness;
- category and context-length breakdowns;
- requested and realized keep budget;
- paired task-level outcomes;
- exact paired McNemar for key comparisons;
- paired/bootstrap confidence intervals;
- exclusions and failures.

Do not treat several turns from the same conversation as independent.

Claim rules:

- Say “matches SnapKV” unless the paired test establishes superiority.
- Do not claim τ-bench task success from retention audit.
- Do not claim model behavior beyond the context window.
- Do not hide FullKV failures; they define the base-model ceiling.
- Keep BFCL results separate from the frozen PriorityBench-A core until the new
  manifest and external evaluation pass their own freeze.

## 14. CPU gold-span audit specification

Freeze extraction rules before looking at policy outcomes.

Candidate span classes:

- tool names and schema fields;
- tool-call arguments;
- identifiers later reused (order, reservation, issue, file, user IDs);
- values from tool results reused later;
- explicit policy lines;
- conversational corrections and superseding constraints.

For each span and policy, report:

- any token retained;
- all tokens retained;
- retained fraction;
- span age;
- visible-structure vs buried/free-form class;
- context length.

Manually audit a random sample of extracted spans and publish precision/error
counts. Do not hand-select only spans favorable to structure.

## 15. Completion gates

### Gate P0 — cluster readiness

- persistent paths and quota verified;
- account/partition/QOS recorded;
- `uv sync --frozen` succeeds;
- model and datasets pinned locally;
- no secrets in Git or logs.

### Gate P1 — harness integrity

- official BFCL scorer passes on known examples;
- one decision point per primary task;
- all arms share byte/token budget;
- real SnapKV path asserted;
- interruption/resume test passes;
- manifest hashes frozen.

### Gate P2 — pilot

- 20-task five-arm paired completion;
- no silent truncation;
- no fake baseline fallback;
- runtime and memory measured;
- final `n=400` or `n=600` chosen before the main run.

### Gate P3 — primary table

- paired completeness at least 95%;
- remaining failures rerun or explicitly reported;
- task-level statistics produced;
- category/length breakdown complete.

### Gate P4 — real-trace audit

- dataset revision frozen;
- extraction rules frozen;
- manual extraction audit complete;
- CPU retention table generated;
- limitations wording included.

### Gate P5 — publication update

- new results added to `RESULTS.md` and `docs/EVIDENCE.md`;
- paper updated without changing old claim boundaries;
- commands/configs/checkpoints documented;
- external-evaluation manifest committed;
- paper figures regenerated from tracked summaries.

## 16. First actions for the Prajna Cursor agent

1. Read the source-of-truth files listed in Section 0.
2. Inspect Prajna with read-only Slurm commands; record but do not guess settings.
3. Search the repository and confirm there is no existing BFCL/τ adapter.
4. Write an implementation plan before coding.
5. Implement dataset normalization, manifest/checkpoint schema, and official
   scoring on CPU.
6. Add unit tests for stable IDs, atomic restart, keep-budget matching,
   context exclusion, and no SnapKV fallback.
7. Add Slurm templates only after discovering Prajna syntax.
8. Run CPU tests.
9. Submit only the one-H100 smoke.
10. Report the smoke output and measured pilot runtime before launching the
    primary array.

## 17. Stop conditions

Stop and report instead of continuing if:

- official BFCL licensing/data access is unavailable;
- model or dataset revision cannot be pinned;
- the H100 partition/account is unknown;
- the lock requires uncontrolled package substitutions;
- SnapKV silently falls back or cannot run on arbitrary BFCL prefixes;
- rendered prompts exceed Qwen context and code tries to truncate them;
- checkpoint restart changes outputs or duplicates points;
- projected primary work exceeds the 72 GPU-hour limit;
- storage use approaches 400 GB.

The agent may reduce the dataset from 600 to 400 only through the predeclared
pilot rule. Any other scientific design change requires a written decision note.
