# Copy-paste prompt for the Cursor agent on Prajna

Replace only the bracketed path/account details if known. The agent must discover
unknown Slurm settings rather than inventing them.

---

```text
You are taking over the PriorityKV external-evaluation work on the Prajna Slurm
cluster.

Repository:
  [ABSOLUTE_PATH_TO_PRIORITY_KV_REPO]

Persistent project root (500 GB quota):
  [ABSOLUTE_PERSISTENT_PRAJNA_ROOT]

Compute envelope:
  - at most 72 TOTAL H100 GPU-hours
  - at most 2 H100s concurrently
  - jobs may be cut off after 7–8 hours
  - Prajna is NOT directly interactive for GPU work
  - every GPU command must be submitted through sbatch
  - use squeue and sacct for monitoring
  - do not run CUDA workloads on the login node

Primary goal:
Make PriorityKV publication-ready by adding:
  (A) a BFCL V3 multi-turn GPU generation table, and
  (B) a CPU-only gold-span retention audit on public τ-bench trajectories.

Do not run τ-bench's live agent loop or user simulator. Do not plan or claim an
end-to-end τ-bench evaluation.

FIRST ACTIONS — before editing or submitting anything:

1. cd to the repository.
2. Read these files in full:
   - docs/PRAJNA_BFCL_HANDOFF.md
   - README.md
   - RESULTS.md
   - docs/EVIDENCE.md
   - docs/REPRODUCIBILITY.md
   - FINAL_RUN_MANIFEST.yaml
   - jobs/README.md
3. Run git status and inspect recent commits.
4. Search the repository for BFCL, tau-bench, OpenHands, Slurm, and sbatch.
5. Verify the handoff's statement that no external-evaluation adapter currently
   exists.
6. Inspect Prajna using read-only commands:
     sinfo
     sinfo -o "%P %a %l %D %G"
     scontrol show partition
     sacctmgr show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null || true
     module avail
   Also inspect local Prajna docs if available.
7. Identify and record:
   - Slurm account/project
   - H100 GPU partition
   - CPU partition
   - QOS
   - correct GPU request syntax
   - maximum wall time
   - persistent 500 GB filesystem
   - whether compute nodes have internet
   - permitted model/dataset download path
   - CUDA module/toolkit path
8. Present a concise implementation plan grounded in what you found, then
   proceed unless a genuine user decision is required.

STRICT SCIENTIFIC RULES:

- Treat existing PriorityBench-A results and FINAL_RUN_MANIFEST.yaml as frozen.
- Do not edit old configs under old job IDs.
- Use new freeze namespace EXTERNAL_BFCL_PRAJNA_V1.
- Pin exact model and dataset revisions.
- Primary model:
    Qwen/Qwen3-8B
    revision b968826d9c46dd6066d109eabc6255188de91218
- Never silently truncate an external prompt.
- Exclude over-context examples with an explicit reason and report them.
- Do not tune the tagger after viewing policy outcomes.
- Do not call DropKeep or another heuristic “SnapKV”.
- Assert that the real attention-based SnapKV path ran.
- All non-FullKV arms must realize the same keep count.
- Use one predeclared decision point per conversation for primary statistics.
- Do not count several turns from one conversation as independent samples.
- Say structure “matches SnapKV” unless paired statistics prove otherwise.
- The τ-bench audit is retention/mechanistic evidence, not task success.
- Never hide FullKV failures or excluded tasks.

RESOURCE RULES:

- Never use more than two H100s at once.
- Prefer one H100 per Slurm array task and cap arrays with %2.
- Do not use tensor parallelism for the 8B model unless a measured blocker
  requires it.
- Never save full KV caches per decision point.
- Stop if persistent storage exceeds 400 GB.
- Use uv and uv.lock only. Do not use ad-hoc pip installs.
- Never print or commit HF_TOKEN or cluster credentials.
- Run CPU parsing/tests before reserving H100s.

IMPLEMENTATION REQUIRED:

Add and test the following components (names may vary slightly, responsibilities
may not):

1. scripts/prepare_bfcl_external.py
   - pin and normalize BFCL V3 multi-turn
   - render with the pinned Qwen tokenizer
   - select one deterministic primary decision point per conversation
   - produce token-length/category reports
   - freeze 600 balanced tasks if feasible:
       100 base
       100 missing parameters
       100 missing functions
       150 long-context
       150 composite
   - minimum allowed after pilot: 400, preserving category balance
   - write exclusions explicitly

2. scripts/run_bfcl_external.py
   - arms: full, structure, uniform, snapkv, random
   - keep_frac=0.25 primary
   - resumable per decision point
   - load the model once per shard
   - process about 25–50 work units per shard
   - atomic result writes
   - skip only validated complete point files
   - catch SIGTERM and flush shard status
   - record timings, memory, keep masks, raw output, parsed call, versions,
     Slurm metadata, and terminal status

3. scripts/score_bfcl_external.py
   - use the official BFCL scorer
   - produce overall/category/length breakdowns
   - schema, function-name, and argument correctness
   - task-level paired outcomes
   - exact paired McNemar and paired/bootstrap confidence intervals
   - paired-completeness and failure reports

4. scripts/audit_tau_retention.py
   - CPU-only public trajectory audit
   - pin AgentSuite/tau-bench-trajectories revision
   - extract frozen span classes:
       schemas/tool names
       tool-call arguments
       reused IDs
       reused tool-result values
       explicit policies
       conversational corrections/superseding constraints
   - report any/all/fraction retained, age, context length, and visible-vs-buried
   - create a manually auditable random extraction sample
   - never invoke a user simulator or perform generation

5. configs/external_bfcl_prajna_v1.yaml
   - all frozen scientific settings

6. Slurm support:
   - cluster/prajna/README.md
   - cluster/prajna/config.example.env
   - cluster/prajna/bootstrap_cpu.sbatch
   - cluster/prajna/smoke_h100.sbatch
   - cluster/prajna/bfcl_array.sbatch
   - cluster/prajna/tau_audit_cpu.sbatch
   Discover site syntax first; do not write guessed partition/account values into
   tracked files. Keep personal values in ignored config.env.

CHECKPOINT CONTRACT:

Work identity must be stable over:
  dataset revision / task ID / decision turn / model revision /
  arm / keep fraction / seed / harness revision

Store:
  $PRAJNA_ROOT/results/external_bfcl_prajna_v1/
    manifest/tasks.jsonl
    manifest/work_items.jsonl
    manifest/exclusions.jsonl
    manifest/hashes.json
    points/<work_id>.json
    failures/<work_id>.json
    shard_logs/
    summaries/

Write <work_id>.json.tmp, flush+fsync, validate, then atomically rename.
On restart, skip only valid final JSON. Never use one giant mutable result JSON.

ENVIRONMENT SETUP:

- Use persistent cache paths under PRAJNA_ROOT:
    UV_CACHE_DIR
    HF_HOME
    TORCH_EXTENSIONS_DIR
    XDG_CACHE_HOME
    PRIORITYKV_SCRATCH
- Reproduce with:
    uv sync --frozen --extra gpu --extra kvpress --extra dev
- Download the pinned Qwen snapshot to:
    $PRIORITYKV_SCRATCH/models/Qwen3-8B
- Prefer a CPU/bootstrap job for environment/model/data staging.
- If compute nodes lack internet, use Prajna's permitted transfer/login workflow.
- Validate model snapshot completeness before using GPU time.

TESTS REQUIRED BEFORE GPU:

- stable work IDs
- deterministic balanced sampling
- context-limit exclusion with no truncation
- exact requested/realized keep-count equality
- explicit assertion against SnapKV fallback
- official scorer integration on known/tiny examples
- atomic checkpoint validation
- restart skips complete points
- restart retries incomplete/corrupt points
- SIGTERM simulation
- paired completeness calculation
- CPU τ-span extraction tests

SLURM EXECUTION GATES:

Gate P0, cluster readiness:
- account/partition/QOS/path/network policy known
- persistent quota checked
- uv locked environment succeeds
- model and datasets pinned locally

Gate P1, one-H100 smoke:
- submit exactly one smoke job
- verify H100 and compute capability 9.0
- import torch/transformers/vllm/flashinfer
- run repository tests
- run scripts/run_flashinfer_packed_parity.py at head_dim=128
- require PARITY_PASS
- run one BFCL point through full, structure, and real SnapKV
- run official scoring
- kill/requeue once and prove resume

Gate P2, 20-task pilot:
- 4 tasks from each BFCL category
- five primary arms
- measure p50/p75 seconds per task-arm and peak memory
- project total GPU-hours
- choose n=600 only if it fits with 20% reserve
- otherwise freeze n=400 with category balance

Gate P3, primary Slurm array:
- one H100 per array task
- no more than %2 concurrent
- shards of 25–50 work units
- <=7.5h task walltime
- keep_frac=0.25
- monitor using squeue/sacct, not interactive GPU commands

Gate P4, stress:
- only after primary paired table is complete
- 200 long-context/composite tasks
- keep_frac=0.10
- arms: full, structure, uniform, snapkv

Gate P5, CPU audit:
- no GPU
- frozen τ dataset and extraction rules
- audit all usable traces or >=1000 stratified
- manually inspect random extracted spans

72 GPU-HOUR BUDGET:
- 6 hours environment/model/JIT smoke
- 8 hours calibration pilot
- 42 hours primary table
- 8 hours stress subset
- 8 hours retries

Track allocated GPU-hours from sacct after every stage. Do not infer cost only
from wall-clock logs. Stop before exceeding 72.

MONITORING:

- Use sbatch --parsable and record every job ID.
- Use squeue for live state.
- Use sacct for terminal state, elapsed time, allocation, MaxRSS, ExitCode.
- Keep stdout/stderr under the persistent project root.
- If a job is clearly invalid, scancel it promptly.
- On timeout/preemption, resubmit only incomplete shards.

FINAL DELIVERABLES:

- external-evaluation manifest with hashes
- pinned BFCL and τ revisions
- 400–600 conversation five-arm BFCL table
- optional 200-task hard-budget table
- CPU τ retention audit
- official scorer output and paired statistics
- failure/exclusion ledger
- environment/version report
- Slurm scripts and exact job IDs
- tracked compact summaries (not raw secrets or giant caches)
- updates to RESULTS.md, docs/EVIDENCE.md, paper manuscript/LaTeX, and figures
- explicit limitations:
    no end-to-end τ-bench rollout
    τ audit is mechanistic only
    over-context exclusions
    FullKV capability ceiling
    structure-vs-SnapKV significance boundary

GIT PRACTICE:

- Do not overwrite frozen configs or result IDs.
- Work in small reviewable commits.
- Run tests before each commit.
- Commit and push the CPU harness + Slurm infrastructure after tests pass.
- Commit result summaries and paper updates separately.
- Never commit .env, cluster/prajna/config.env, HF tokens, model files, raw caches,
  or oversized logs.

AUTONOMY:

Proceed through implementation, CPU tests, environment bootstrap, one-H100 smoke,
and pilot. If the pilot passes and the projected design fits <=72 GPU-hours with
20% reserve, submit the primary array. Ask me only when:

- account/partition/access is unavailable,
- a dataset/model license requires my action,
- a scientific design choice would change the frozen protocol,
- SnapKV cannot run without substituting another method,
- projected work cannot fit the budget, or
- a destructive/irreversible action is required.

At each gate, report:
- what passed/failed,
- Git commit,
- Slurm job IDs,
- GPU-hours consumed and remaining,
- completed/failed/missing point counts,
- next action.

Do not merely write a plan. Implement the CPU-safe pieces first, verify them, and
then use Slurm according to the gates above.
```

---

## Optional user-supplied details

Add these above the prompt if already known:

```text
Prajna account:
GPU partition:
CPU partition:
QOS:
Persistent root:
CUDA module:
GPU request syntax:
Maximum Slurm wall time:
Compute-node internet available: yes/no
```
