#!/usr/bin/env python3
"""Dual-GPU driver for d4_fp8_compare: 8k ∥ 16k (max 2 GPUs)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _merge(a: dict, b: dict) -> dict:
    rows = list(a.get("rows") or []) + list(b.get("rows") or [])
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def _mean(xs):
        vals = [float(x) for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    def _summ(rs):
        return {
            "n": len(rs),
            "score_mean": _mean([r.get("score") for r in rs]),
            "e2e_ttft_ms_mean": _mean([r.get("e2e_ttft_ms") for r in rs]),
            "tpot_ms_mean": _mean([r.get("tpot_ms") for r in rs]),
            "pack_ms_mean": _mean([r.get("pack_ms") for r in rs]),
        }

    arms = {k: _summ(v) for k, v in by_arm.items()}
    by_ctx: dict[str, dict] = {}
    for r in rows:
        ctx = str(int(r.get("context_length") or 0))
        by_ctx.setdefault(ctx, {}).setdefault(r["arm"], []).append(r)
    summary_by_ctx = {c: {a: _summ(rs) for a, rs in arms.items()} for c, arms in by_ctx.items()}
    ok = bool(a.get("pass")) and bool(b.get("pass"))
    return {
        "job": "d4_fp8_compare_dual",
        "decision": "D4_FP8_COMPARE_PASS" if ok else "D4_FP8_COMPARE_PARTIAL",
        "pass": ok,
        "arms": arms,
        "by_context": summary_by_ctx,
        "n_examples": len({r.get("example_id") for r in rows}),
        "rows": rows,
        "note": "dual merge 8k∥16k",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "d4_fp8_compare.yaml"))
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "d4_fp8_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d4_fp8_compare_{args.out_tag}.json"

    ids = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    ids = ids[:2]
    contexts = ["8000", "16000"]
    if len(ids) < 2:
        ids = [ids[0] if ids else "0"]
        # sequential
        codes = []
        shards = []
        t0 = time.time()
        for ctx in contexts:
            tag = f"{args.out_tag}_c{ctx}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ids[0]
            env["PRIORITYKV_ONLY_CONTEXT_LENGTH"] = ctx
            sp = out_dir / f"d4_fp8_compare_{tag}.json"
            if sp.exists():
                sp.unlink()
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_d4_fp8_compare.py"),
                "--config",
                args.config,
                "--out-tag",
                tag,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
            ]
            log = out_dir / f"dual_{tag}.log"
            with open(log, "w", encoding="utf-8") as lf:
                codes.append(subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT)))
            shards.append(json.loads(sp.read_text()) if sp.exists() else {"pass": False})
        merged = _merge(shards[0], shards[1]) if len(shards) == 2 else {"pass": False, "decision": "D4_FP8_COMPARE_PARTIAL"}
        merged["seconds"] = round(time.time() - t0, 3)
        merged["exit_codes"] = codes
        merged["tag"] = args.out_tag
        out_path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
        print(json.dumps({"decision": merged.get("decision"), "pass": merged.get("pass")}, indent=2))
        print(f"out={out_path}")
        return 0 if merged.get("pass") else 1

    t0 = time.time()
    procs = []
    shard_paths = []
    for gpu, ctx in zip(ids, contexts, strict=True):
        tag = f"{args.out_tag}_c{ctx}"
        sp = out_dir / f"d4_fp8_compare_{tag}.json"
        if sp.exists():
            sp.unlink()
        shard_paths.append(sp)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PRIORITYKV_ONLY_CONTEXT_LENGTH"] = ctx
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_d4_fp8_compare.py"),
            "--config",
            args.config,
            "--out-tag",
            tag,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        log = out_dir / f"dual_{tag}.log"
        lf = open(log, "w", encoding="utf-8")
        print(f"[dual-fp8] start ctx={ctx} gpu={gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT)))
        procs[-1]._log = lf  # type: ignore[attr-defined]

    codes = []
    for p in procs:
        codes.append(p.wait())
        p._log.close()  # type: ignore[attr-defined]

    if any(c != 0 for c in codes) or any(not p.exists() for p in shard_paths):
        # Still try merge if files exist
        pass
    shards = [json.loads(p.read_text()) for p in shard_paths if p.exists()]
    if len(shards) < 2:
        result = {"decision": "D4_FP8_COMPARE_PARTIAL", "pass": False, "exit_codes": codes}
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 1
    merged = _merge(shards[0], shards[1])
    merged["seconds"] = round(time.time() - t0, 3)
    merged["exit_codes"] = codes
    merged["tag"] = args.out_tag
    out_path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
    print(json.dumps({"decision": merged["decision"], "pass": merged["pass"], "arms": merged["arms"]}, indent=2))
    print(f"out={out_path}")
    return 0 if merged["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
