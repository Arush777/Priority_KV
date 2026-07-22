# Prajna Cluster Notes

Living reference for working on the Prajna HPC cluster (IIT Bombay).
**Keep this updated as new cluster facts are discovered.**

Last updated: 2026-07-21

---

## Environment

| Property | Value |
|---|---|
| Cluster | `prajna.iitb.ac.in` |
| OS | Rocky Linux 8.10 (Green Obsidian) |
| Kernel | 4.18.0-553.el8_10.x86_64 |
| Scheduler | Slurm (`/opt/slurm/bin`) |
| Home | `/home/medal/anupam.rawat` on Lustre (1.3P total, ~51% used) |
| Spack | `/lustre-flash/apps/spack` |
| Conda | `~/miniconda3` |
| Root access | **No sudo** (password required, none available) |

### Filesystem note
Home is on Lustre. First access to a newly written file can be **slow** — commands
that touch fresh files may take >2 min on first run, then are fast. This is normal
metadata latency, not a hang. Don't kill and retry; wait it out.

---

## Login nodes

`prajna.iitb.ac.in` is **round-robin DNS** across two login nodes. You get a random
one on each connection.

| Host | IP | DNS status |
|---|---|---|
| `login1.prajna.iitb.ac.in` | `10.195.100.108` | ✅ resolves correctly |
| `login2.prajna.iitb.ac.in` | `10.195.100.109` | ⚠️ **broken** — resolves only to link-local IPv6 (`fe80::…`), unroutable |
| `login.prajna.iitb.ac.in` | `10.195.100.101` | separate host |

### ⚠️ Pinning to a specific login node

tmux sessions are **per-node** (socket lives in node-local `/tmp/tmux-$UID/`). If you
start a session on login2 and reconnect onto login1, it is invisible. You must return
to the same node.

Because `login2`'s DNS name is broken, **use the raw IP**. Put this in `~/.ssh/config`
on your **laptop** (not on the cluster):

```sshconfig
Host login2
    HostName 10.195.100.109
    User anupam.rawat
    ServerAliveInterval 60
    ServerAliveCountMax 10

Host login1
    HostName 10.195.100.108
    User anupam.rawat
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

Then `ssh login2`. The `ServerAliveInterval` lines prevent idle NAT timeouts from
dropping the connection during long runs.

From *inside* the cluster, `ssh login2` works via `/etc/hosts` (short names are
mapped correctly on the nodes themselves).

**Network caveat:** all addresses are RFC1918 private; client connects from
`10.204.8.1`. This assumes campus network or IITB VPN. From outside, a bastion +
`ProxyJump` would be needed — untested.

---

## tmux

**Installed 2026-07-21.** Not available system-wide; installed per-user via conda.

```bash
conda create -y -n tools -c conda-forge tmux
ln -sf ~/miniconda3/envs/tools/bin/tmux ~/bin/tmux
```

- Version: **tmux 3.7**
- Binary: `~/miniconda3/envs/tools/bin/tmux`, symlinked to `~/bin/tmux`
- `~/bin` is already on `PATH` — no shell rc edit was needed
- Verified: server starts, session lists, session kills. Working.

Installed into a dedicated `tools` env rather than `base` to keep `base` clean.

### Why not other routes
- `yum`/`dnf` — no root, and repos are broken anyway (`ookla_speedtest-cli` repo
  throws SSL errors that abort metadata refresh)
- `spack` — has `tmux`, viable fallback, but builds from source (slow)
- `screen`, `zellij` — not installed either

---

## Running Claude Code in tmux

Verified safe on this cluster. The usual killers were checked:

| Risk | Status |
|---|---|
| `KillUserProcesses` in logind | ✅ `no` — processes survive logout |
| Cron process reapers | ✅ none (`/etc/cron.d`: only `0hourly`, `raid-check`) |
| API egress | ✅ `api.anthropic.com` reachable (HTTP 405 to bare GET = expected) |
| Node runtime | ✅ `claude` is a native binary at `~/.local/bin/claude`; no `node` needed and none installed |
| Process limit | ✅ `ulimit -u` = 386432 |

### Caveats
1. **`Linger=no`.** Harmless while `KillUserProcesses=no`, but if an admin ever flips
   the kill policy, detached sessions start dying. `loginctl enable-linger` would
   harden this — needs root, so it's a sysadmin ask.
2. **Never run the long-lived session inside a Slurm job.** tmux under `srun`/`sbatch`
   dies at the job time limit regardless of detach. Keep tmux on the *login node* and
   let it submit work outward with `sbatch`.
3. **Login-node etiquette.** Claude Code itself is light, but anything heavy it spawns
   belongs in Slurm.

---

## tmux quick reference

Prefix is **`Ctrl-b`** (press and release, *then* the next key).

### Sessions
```bash
tmux new -s work        # create named session
tmux ls                 # list sessions
tmux attach -t work     # reattach
tmux kill-session -t work
```

| Keys | Action |
|---|---|
| `Ctrl-b` `d` | **detach** (leaves everything running) |
| `Ctrl-b` `s` | interactive session picker |
| `Ctrl-b` `$` | rename session |

### Windows (tabs)
| Keys | Action |
|---|---|
| `Ctrl-b` `c` | new window |
| `Ctrl-b` `n` / `p` | next / previous window |
| `Ctrl-b` `0`–`9` | jump to window N |
| `Ctrl-b` `,` | rename window |
| `Ctrl-b` `&` | kill window |

### Panes (splits)
| Keys | Action |
|---|---|
| `Ctrl-b` `%` | split vertical (left/right) |
| `Ctrl-b` `"` | split horizontal (top/bottom) |
| `Ctrl-b` arrow | move between panes |
| `Ctrl-b` `z` | zoom/unzoom current pane |
| `Ctrl-b` `x` | kill pane |

### Scrolling
`Ctrl-b` `[` enters copy/scroll mode — arrows / PgUp to scroll, `q` to exit.
This is the one people trip on: you **cannot** scroll back with the mouse wheel by
default, you must enter copy mode first.

### Typical Claude Code workflow
```bash
ssh login2                       # the IP-pinned alias
tmux new -s claude               # or: tmux attach -t claude
cd ~/Priority_KV
claude
# Ctrl-b d to detach; close laptop; come back later
ssh login2 && tmux attach -t claude
```

### Optional quality-of-life
Not applied — `~/.tmux.conf` does not currently exist. If wanted:
```tmux
set -g mouse on              # mouse scroll + pane select
set -g history-limit 50000   # deeper scrollback
set -g status-bg colour234
set -g status-fg white
```

---

## GPU partitions, QOS, and CUDA (discovered 2026-07-21)

**There is no H100 and no sm_90 device anywhere on Prajna.**

| Partition | Required QOS | GPU | Cap | Max GPU/user | Jobs/user | Max wall |
|---|---|---|---|---:|---:|---|
| `dgx` | `dgx` | 8/node, A100-class | 8.0 | 4 | 4 | 6 d |
| `a40` | `a40` | A40 48 GB | 8.6 | 2 | 3 | 4 d |
| `l40` *(default)* | `l40` | L40S 46 GB | 8.9 | 4 | 4 | 2 d |
| `interactive` | `interactive` | mixed | — | 8 | 2 | 4 h |
| `debug` | `debug` | A40 | 8.6 | — | — | 30 min |

- **The partition's QOS is mandatory.** Without `--qos`, submission fails with
  `Invalid qos specification`. No account string is needed for this association.
- `dgx` is frequently **100% allocated** (all 72 GPUs); `l40` and `a40` schedule
  in under a minute. Prefer `l40` for single-GPU work.
- Driver is **570.86.15 = CUDA 12.8**. `~/job.slurm` already recorded this as
  `dgx(12.4_dgx), a40(12.8), l40(12.8)` — check it before picking a torch build.

### Gotchas that cost real time

1. **PyPI torch is now built for CUDA 13** and aborts with "NVIDIA driver is too
   old". Pin the cu128 index. See [[uv-sources-direct-deps-only]] — `[tool.uv.sources]`
   only redirects *direct* deps, so transitive `torchvision`/`torchaudio` silently
   stay on the CUDA 13 build and break `import transformers` with
   `RuntimeError: operator torchvision::nms does not exist`.
2. **No CUDA toolkit exists on the cluster.** No `nvcc` on compute nodes, spack has
   no cuda build, `/usr/local/cuda` is an empty stub. The pip
   `nvidia-cuda-nvcc-cu12` package ships only `ptxas`, not the `nvcc` driver, so it
   cannot compile FlashInfer's JIT kernels. What works:
   `conda create -p <prefix> -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-crt=12.8`,
   then symlink `targets/x86_64-linux/{include,lib}` into `<prefix>/{include,lib64}`
   so the classic `$CUDA_HOME` layout exists, and export `CUDA_HOME`.
3. **Compute nodes have no DNS or outbound network.** Login nodes do. Stage every
   model, dataset, and wheel from the login node and reference it by local path.
   Set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so failures are loud.
4. **`/tmp` is node-local.** A file written to `/tmp` on the login node is invisible
   to the compute node. Runtime configs and data must live under `$HOME`.
5. **Only `$HOME` is writable.** `/lustre-scratch`, `/lustre-flash`, and `/scratch`
   all reject user writes, despite having hundreds of TB free.
6. `lfs quota` reports limit 0 (no enforced quota) with ~613 TB free, but `$HOME`
   was already at ~473 GB, 241 GB of it an unrelated `~/.cache/huggingface`.
