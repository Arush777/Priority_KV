# PriorityKV-Agent

Structure-protected mixed-precision KV cache (BF16/INT4) for long multi-turn agent traces.
Solo project. Primary hardware: H200.

**Headline claim:** uniform KV compression silently breaks tool schemas / instruction hierarchies in long agent traces even when average accuracy is flat; PriorityKV removes those failures at ~30% of FullKV bytes with measured H200 serving gains.

The full research plan (scope, gates, week plan) is in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## Dual-machine workflow (read this first)

| Where | Role |
|---|---|
| **This machine** (Cursor agents) | Write code, CPU tests, commit, **push to GitHub** |
| **Your H200** (you only) | `git pull`, `uv` env, download models, **run GPU work** |

Agents never run on the H200. You operate that box manually.

Exact H200 commands: [`docs/H200_SETUP.md`](docs/H200_SETUP.md).

---

## Repo layout

```
src/prioritybench/   # PriorityBench-A generator + deterministic scorers
src/prioritykv/      # byte model → page manager → INT4 → mixed attention (build in order)
scripts/             # CLI + smoke tests
tests/               # unit tests (CPU)
configs/             # frozen run configs / manifests
data/prioritybench/  # fixtures tracked; generated splits gitignored
docs/                # plan + H200 setup
```

---

## First steps (agent machine — do now)

```bash
cd /u/arushh/Arush/Priority_KV
./scripts/sync.sh
git push origin main
```

## First steps (H200 — you)

```bash
git clone git@github.com:Arush777/Priority_KV.git
cd Priority_KV
./scripts/sync.sh            # CPU deps + checks
# edit .env: REPO_ROOT, scratch, HF_*, CUDA_VISIBLE_DEVICES=6,7
./scripts/sync.sh --cuda     # GPU stack; capped to two devices via .env
```

See [`docs/H200_SETUP.md`](docs/H200_SETUP.md).

---

## Week 0 checklist (solo)

- [ ] Repo pushes cleanly; H200 can `git pull`
- [ ] `./scripts/smoke_cpu.sh` green here and on H200
- [ ] H200: `uv sync --extra gpu` + `torch.cuda.is_available() == True`
- [ ] Qwen3-8B pinned revision downloaded to scratch
- [ ] Gemma secondary release/license verified (note in `docs/decisions.md`)
- [ ] Live workshop CFP picked for D6
