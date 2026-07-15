#!/usr/bin/env python3
"""Wait until N physical GPUs have enough free VRAM, then exec a command.

On the shared 8×H200 host, our allocated pair (often 6,7) may be full while
another pair is free. This helper scans *all* GPUs via nvidia-smi, waits until
``--num-gpus`` cards each have ``--min-free-gib`` free, then sets
``CUDA_VISIBLE_DEVICES`` to those indices and runs the command.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def gpu_free_gib() -> list[tuple[int, float]] | None:
    """Return [(physical_index, free_gib), ...] for every GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[wait_gpu] nvidia-smi failed: {exc}", flush=True)
        return None
    rows: list[tuple[int, float]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        rows.append((int(parts[0]), float(parts[1]) / 1024.0))
    return rows


def pick_gpus(
    rows: list[tuple[int, float]], min_free_gib: float, num_gpus: int
) -> list[int]:
    eligible = sorted(
        (idx for idx, free in rows if free >= min_free_gib),
        key=lambda i: i,
    )
    return eligible[:num_gpus]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gib", type=float, default=95.0)
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--poll-sec", type=float, default=60.0)
    ap.add_argument("--timeout-sec", type=float, default=0.0, help="0 = wait forever")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = ap.parse_args()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[wait_gpu] missing command after --", file=sys.stderr)
        return 2
    if args.num_gpus < 1:
        print("[wait_gpu] --num-gpus must be >= 1", file=sys.stderr)
        return 2

    t0 = time.time()
    print(
        f"[wait_gpu] need {args.num_gpus} GPU(s) with >={args.min_free_gib:.1f} GiB free "
        f"(any physical indices); poll={args.poll_sec}s cmd={' '.join(cmd)}",
        flush=True,
    )
    chosen: list[int] = []
    while True:
        rows = gpu_free_gib()
        if rows is not None:
            snapshot = " ".join(f"{i}:{f:.1f}GiB" for i, f in rows)
            chosen = pick_gpus(rows, args.min_free_gib, args.num_gpus)
            if len(chosen) >= args.num_gpus:
                print(
                    f"[wait_gpu] ready GPUs={chosen} snapshot=[{snapshot}] — starting",
                    flush=True,
                )
                break
        elapsed = time.time() - t0
        if args.timeout_sec > 0 and elapsed >= args.timeout_sec:
            print(
                f"[wait_gpu] timeout after {elapsed:.0f}s "
                f"(last_chosen={chosen} last={rows})",
                file=sys.stderr,
                flush=True,
            )
            return 3
        print(
            f"[wait_gpu] waiting need={args.num_gpus}x>={args.min_free_gib} "
            f"eligible={chosen} snapshot=[{snapshot if rows else 'n/a'}] "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )
        time.sleep(args.poll_sec)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in chosen)
    print(f"[wait_gpu] export CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
