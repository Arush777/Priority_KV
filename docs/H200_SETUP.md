# H200 / GPU server setup — PriorityKV (solo)

You write and push code from the **agent machine** (`/u/arushh/Arush/Priority_KV`).
You **only pull and run** on the H200. Never expect Cursor agents on the H200.

## Split of duties

| Machine | You do |
|---|---|
| Agent machine (here) | edit code, `uv sync --extra dev`, CPU smoke, `git push` |
| H200 | `git pull`, `./scripts/setup_env.sh --gpu`, download models, run evals |

---

## Step 0 — One-time on H200: tools

SSH into the H200, then:

```bash
# Optional but recommended: put heavy stuff on fast local scratch
export SCRATCH="${SCRATCH:-$HOME/scratch}"   # change if your lab uses /scratch/$USER etc.
mkdir -p "$SCRATCH"/{models,datasets,runs,hf_cache}

# uv (Python env manager — lockfile ships in the repo)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Sanity: you need a working NVIDIA driver + CUDA for later steps
nvidia-smi
```

If `nvidia-smi` fails, stop and fix drivers before installing torch/vLLM.

---

## Step 1 — First thing: get the repo (clone once)

```bash
cd "$SCRATCH"   # or wherever you want the code; avoid tiny home quotas if you have them
git clone git@github.com:Arush777/Priority_KV.git
cd Priority_KV
```

If you already cloned once, forever after:

```bash
cd /path/to/Priority_KV
git pull origin main
```

Use SSH (`git@github.com:...`) if you have a deploy key / ssh key on the H200.
HTTPS alternative:

```bash
git clone https://github.com/Arush777/Priority_KV.git
```

---

## Step 2 — Create the uv environment (CPU smoke first)

```bash
cd /path/to/Priority_KV
./scripts/setup_env.sh
```

That will:
1. `uv sync --extra dev` (creates `.venv`, installs CPU deps from the lockfile)
2. run `./scripts/smoke_cpu.sh` (byte-model + PriorityBench unit tests)

Edit `.env` after first run:

```bash
nano .env
```

Set at least:

```bash
REPO_ROOT=/absolute/path/to/Priority_KV
PRIORITYKV_SCRATCH=/absolute/path/to/scratch   # models + outputs
HF_TOKEN=hf_...                                # from https://huggingface.co/settings/tokens
HF_HOME=/absolute/path/to/scratch/hf_cache
```

Then:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## Step 3 — GPU extras (torch / transformers / vLLM)

Only on a node where `nvidia-smi` works:

```bash
cd /path/to/Priority_KV
./scripts/setup_env.sh --gpu
```

Verify:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect something like `True` and `NVIDIA H200`.

---

## Step 4 — Download the primary model (Qwen3-8B)

Pinned revision is in `src/prioritybench/pins.py`.

```bash
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PRIORITYKV_SCRATCH/hf_cache}"
export HF_TOKEN  # from .env

huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$PRIORITYKV_SCRATCH/models/Qwen3-8B"
```

Do **not** commit the model weights. They stay under scratch.

---

## Day-to-day loop

**On agent machine (me / Cursor):**

```bash
# after code changes
./scripts/smoke_cpu.sh
git add -A && git commit -m "..." && git push origin main
```

**On H200 (you):**

```bash
cd /path/to/Priority_KV
git pull origin main
./scripts/setup_env.sh --gpu   # only if pyproject/lock changed
# then run whatever script the agent told you (eval / vLLM / etc.)
```

---

## What NOT to do

- Do not run GPU / vLLM on the agent/login machine.
- Do not `git add` `models/`, `.venv/`, or `.env`.
- Do not expect agents to SSH into the H200.
