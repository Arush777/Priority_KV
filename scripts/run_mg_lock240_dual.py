#!/usr/bin/env python3
"""Dual-GPU lock-240 quality (max 2 GPUs): 8k∥16k then 32k.

Hard cap: never use more than 2 visible GPUs.
Wave 1: context 8k ∥ 16k on the two GPUs.
Wave 2: context 32k on GPU 0 (sequential).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _merge_shards(shards: list[dict[str, Any]], tag: str) -> dict[str, Any]:
    policies = list(shards[0].get("policies") or ["full", "uniform", "structure"])
    all_ids: list[str] = []
    by_ctx: dict[str, dict[str, Any]] = {}
    arm_scores: dict[str, list[float]] = {p: [] for p in policies}
    arm_int4: dict[str, list[float]] = {p: [] for p in policies}
    arm_bytes: dict[str, list[float]] = {p: [] for p in policies}
    seconds: dict[str, float] = {}

    for sh in shards:
        all_ids.extend(sh.get("selected_ids") or [])
        for p in policies:
            arm = (sh.get("arms_detail") or sh.get("arms") or {}).get(p) or {}
            rows = arm.get("rows") or []
            if rows:
                for row in rows:
                    arm_scores[p].append(float(row["policy_score"]))
                    meta = row.get("meta") or {}
                    if "int4_frac_realized" in meta:
                        arm_int4[p].append(float(meta["int4_frac_realized"]))
                    if "payload_bytes" in meta:
                        arm_bytes[p].append(float(meta["payload_bytes"]))
            elif "mean" in arm:
                n = int(sh.get("n") or 0)
                if n and arm.get("mean") is not None:
                    arm_scores[p].extend([float(arm["mean"])] * n)
                if arm.get("int4_frac_realized") is not None and n:
                    arm_int4[p].extend([float(arm["int4_frac_realized"])] * n)
            for sec_k, sec_v in (sh.get("seconds") or {}).items():
                seconds[f"{sh.get('shard_ctx', '?')}:{sec_k}"] = float(sec_v)

        ctxs = sorted(
            {
                int(r.get("context_length", 0))
                for p in policies
                for r in ((sh.get("arms_detail") or {}).get(p) or {}).get("rows") or []
            }
        )
        for L in ctxs:
            key = str(L)
            by_ctx.setdefault(key, {p: [] for p in policies})
            for p in policies:
                rows = ((sh.get("arms_detail") or {}).get(p) or {}).get("rows") or []
                for row in rows:
                    if int(row.get("context_length", -1)) == L:
                        by_ctx[key][p].append(float(row["policy_score"]))

    full_mean = _mean(arm_scores.get("full") or [])
    arms_out: dict[str, Any] = {}
    for p in policies:
        mean = _mean(arm_scores[p])
        arms_out[p] = {
            "mean": mean,
            "n": len(arm_scores[p]),
            "int4_frac_realized": _mean(arm_int4[p]),
            "payload_bytes_mean": _mean(arm_bytes[p]),
            "delta_minus_full": (
                None if mean is None or full_mean is None else mean - full_mean
            ),
        }

    by_context = {
        ctx: {p: {"n": len(xs), "mean": _mean(xs)} for p, xs in arms.items()}
        for ctx, arms in by_ctx.items()
    }

    n_unique = len(set(all_ids))
    ok = n_unique == 240 and all(arms_out[p]["n"] == 240 for p in policies)
    decision = "MG_LOCK240_PASS" if ok else "MG_LOCK240_PARTIAL"
    return {
        "job": "mg_lock240_quality_dual",
        "tag": tag,
        "decision": decision,
        "pass": ok,
        "n": n_unique,
        "policies": policies,
        "fullkv_mean": full_mean,
        "arms": arms_out,
        "by_context": by_context,
        "seconds_by_shard": seconds,
        "selected_ids": sorted(set(all_ids)),
        "note": "dual-GPU max: wave1 8k∥16k, wave2 32k",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mg_lock240_quality.yaml"))
    ap.add_argument("--out-tag", default="r1")
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "mixed_serve"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mg_lock240_quality_{args.out_tag}.json"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ids = [x.strip() for x in visible.split(",") if x.strip() != ""][:2]
    if not ids:
        ids = ["0"]

    t0 = time.time()
    shard_paths: list[tuple[str, Path]] = []
    codes: list[int] = []

    def _run_one(gpu: str, ctx: str) -> int:
        tag = f"{args.out_tag}_c{ctx}"
        shard_out = out_dir / f"mg_lock240_quality_{tag}.json"
        if shard_out.exists():
            shard_out.unlink()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PRIORITYKV_ONLY_CONTEXT_LENGTH"] = ctx
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_mixed_serve.py"),
            "--config",
            args.config,
            "--out",
            str(shard_out),
        ]
        log_path = out_dir / f"dual_{tag}.log"
        print(f"[dual] start ctx={ctx} gpu={gpu} out={shard_out}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log_f:
            rc = subprocess.call(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ROOT)
            )
        print(f"[dual] done tag={tag} exit={rc} log={log_path}", flush=True)
        shard_paths.append((ctx, shard_out))
        return rc

    def _run_parallel(pairs: list[tuple[str, str]]) -> list[int]:
        procs = []
        local_codes: list[int] = []
        for gpu, ctx in pairs:
            tag = f"{args.out_tag}_c{ctx}"
            shard_out = out_dir / f"mg_lock240_quality_{tag}.json"
            if shard_out.exists():
                shard_out.unlink()
            shard_paths.append((ctx, shard_out))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PRIORITYKV_ONLY_CONTEXT_LENGTH"] = ctx
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_mixed_serve.py"),
                "--config",
                args.config,
                "--out",
                str(shard_out),
            ]
            log_path = out_dir / f"dual_{tag}.log"
            log_f = open(log_path, "w", encoding="utf-8")
            print(f"[dual] start ctx={ctx} gpu={gpu} out={shard_out}", flush=True)
            procs.append(
                (
                    tag,
                    subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        cwd=str(ROOT),
                    ),
                    log_f,
                    log_path,
                )
            )
        for tag, p, log_f, log_path in procs:
            rc = p.wait()
            log_f.close()
            local_codes.append(rc)
            print(f"[dual] done tag={tag} exit={rc} log={log_path}", flush=True)
        return local_codes

    # Wave 1: 8k ∥ 16k (1 GPU each if 2 available; else sequential).
    if len(ids) >= 2:
        codes.extend(_run_parallel([(ids[0], "8000"), (ids[1], "16000")]))
    else:
        codes.append(_run_one(ids[0], "8000"))
        codes.append(_run_one(ids[0], "16000"))

    # Wave 2: 32k on first GPU only.
    codes.append(_run_one(ids[0], "32000"))

    if any(c != 0 for c in codes):
        result = {
            "decision": "MG_LOCK240_FAIL",
            "pass": False,
            "error": "one or more shard processes exited nonzero",
            "exit_codes": codes,
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 1

    shards: list[dict[str, Any]] = []
    for ctx, path in shard_paths:
        if not path.exists():
            result = {
                "decision": "MG_LOCK240_FAIL",
                "pass": False,
                "error": f"missing shard {path}",
                "exit_codes": codes,
            }
            out_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            print(f"out={out_path}")
            return 1
        sh = json.loads(path.read_text(encoding="utf-8"))
        sh["shard_ctx"] = ctx
        shards.append(sh)

    merged = _merge_shards(shards, args.out_tag)
    merged["seconds"] = round(time.time() - t0, 3)
    merged["exit_codes"] = codes
    merged["gpus_used"] = ids
    out_path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": merged["decision"],
                "pass": merged["pass"],
                "n": merged["n"],
                "gpus_used": ids,
                "arms": {
                    p: {
                        "mean": v.get("mean"),
                        "delta_minus_full": v.get("delta_minus_full"),
                        "int4_frac_realized": v.get("int4_frac_realized"),
                    }
                    for p, v in merged["arms"].items()
                },
                "by_context": merged["by_context"],
                "seconds": merged["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if merged["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
