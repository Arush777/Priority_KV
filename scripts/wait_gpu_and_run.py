#!/usr/bin/env python3
"""Wait until the primary visible GPU has enough free VRAM, then exec a command.

Shared H200 often has ~25–30 GiB free on CUDA_VISIBLE_DEVICES while another
tenant holds the rest. vLLM will pass a low gpu_memory_utilization check and
still OOM during KV init. Prefer waiting for a real free GPU over scavenging.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def free_gib_device0() -> float | None:
    env = os.environ.copy()
    # Respect CUDA_VISIBLE_DEVICES already set by the worker.
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            env=env,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[wait_gpu] nvidia-smi failed: {exc}", flush=True)
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None
    # First line = visible device 0 after CUDA_VISIBLE_DEVICES remap.
    return float(lines[0]) / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gib", type=float, default=95.0)
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

    t0 = time.time()
    print(
        f"[wait_gpu] need>={args.min_free_gib:.1f} GiB free on visible GPU0; "
        f"poll={args.poll_sec}s cmd={' '.join(cmd)}",
        flush=True,
    )
    while True:
        free = free_gib_device0()
        if free is not None and free >= args.min_free_gib:
            print(f"[wait_gpu] ready free={free:.2f} GiB — starting", flush=True)
            break
        elapsed = time.time() - t0
        if args.timeout_sec > 0 and elapsed >= args.timeout_sec:
            print(
                f"[wait_gpu] timeout after {elapsed:.0f}s "
                f"(last_free={free})",
                file=sys.stderr,
                flush=True,
            )
            return 3
        print(
            f"[wait_gpu] waiting free={free} GiB need>={args.min_free_gib} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )
        time.sleep(args.poll_sec)

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
