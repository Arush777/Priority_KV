#!/usr/bin/env python3
"""Dual-GPU D4 latency driver: run 8k and 16k shards in parallel.

Worker is single-threaded, so one job with CUDA_VISIBLE_DEVICES=5,6
spawns two children (cuda:0 / cuda:1) and merges summaries + M3 gates.
"""

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
    by_ctx = {}
    by_ctx.update(a.get("by_context") or {})
    by_ctx.update(b.get("by_context") or {})
    # Recompute global arms from rows.
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def _mean(xs):
        vals = [float(x) for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    def _arm_summary(rs):
        return {
            "n": len(rs),
            "prefill_ms_mean": _mean([r.get("prefill_ms") for r in rs]),
            "pack_ms_mean": _mean([r.get("pack_ms") for r in rs]),
            "cold_scratch_ms_mean": _mean([r.get("cold_scratch_ms") for r in rs]),
            "decode_ttft_ms_mean": _mean([r.get("decode_ttft_ms") for r in rs]),
            "e2e_ttft_ms_mean": _mean([r.get("e2e_ttft_ms") for r in rs]),
            "tpot_ms_mean": _mean([r.get("tpot_ms") for r in rs]),
            "tokens_per_s_mean": _mean([r.get("tokens_per_s") for r in rs]),
            "score_mean": _mean([r.get("score") for r in rs]),
            "int4_tokens_mean": _mean([r.get("int4_tokens") for r in rs]),
            "payload_bytes_mean": _mean([r.get("payload_bytes") for r in rs]),
        }

    arms = {arm: _arm_summary(rs) for arm, rs in by_arm.items()}
    # Prefer child's m3 ctx_gates merged.
    m3_gates = {}
    for src in (a, b):
        m3_gates.update((src.get("m3") or {}).get("ctx_gates") or {})
    all_ok = all(bool(g.get("pass")) for g in m3_gates.values()) if m3_gates else False
    no_mat = all(not r.get("used_materialize_hf_past") for r in rows)
    if no_mat and all_ok:
        decision = "D4_M3_PASS"
    elif no_mat:
        decision = "D4_M3_GATE_FAIL"
    else:
        decision = "D4_M3_FAIL"
    return {
        "job": "d4_latency_m3_dual",
        "tag": a.get("tag") or b.get("tag"),
        "decision": decision,
        "pass": decision == "D4_M3_PASS",
        "m3_gate": True,
        "m3": {"ctx_gates": m3_gates, "all_ok": all_ok},
        "shards": {
            "a": {"decision": a.get("decision"), "seconds": a.get("seconds"), "n": a.get("n_examples")},
            "b": {"decision": b.get("decision"), "seconds": b.get("seconds"), "n": b.get("n_examples")},
        },
        "n_examples": len({r.get("example_id") for r in rows}),
        "arms": arms,
        "by_context": by_ctx,
        "cuda_peak_bytes": max(int(a.get("cuda_peak_bytes") or 0), int(b.get("cuda_peak_bytes") or 0)),
        "device": f"{a.get('device')} + {b.get('device')}",
        "seconds": None,  # filled by caller
        "rows": rows,
        "note": "dual-GPU shard merge (8k ∥ 16k)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--e2e-gate-mult", type=float, default=1.15)
    ap.add_argument("--ttft-gate-mult", type=float, default=1.25)
    ap.add_argument("--tpot-gate-mult", type=float, default=1.25)
    ap.add_argument("--pack-ms-max", type=float, default=200.0)
    ap.add_argument("--cold-ms-max", type=float, default=100.0)
    ap.add_argument("--pack-ms-max-16k", type=float, default=400.0)
    ap.add_argument("--cold-ms-max-16k", type=float, default=200.0)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "d4_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d4_latency_{args.out_tag}.json"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
    if len(ids) < 2:
        print(f"[dual] only {len(ids)} GPU(s) visible ({visible}); falling back to single run", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_d4_latency.py"),
            "--config",
            args.config,
            "--out-tag",
            args.out_tag,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--m3-gate",
            "--e2e-gate-mult",
            str(args.e2e_gate_mult),
            "--ttft-gate-mult",
            str(args.ttft_gate_mult),
            "--tpot-gate-mult",
            str(args.tpot_gate_mult),
            "--pack-ms-max",
            str(args.pack_ms_max),
            "--cold-ms-max",
            str(args.cold_ms_max),
            "--pack-ms-max-16k",
            str(args.pack_ms_max_16k),
            "--cold-ms-max-16k",
            str(args.cold_ms_max_16k),
        ]
        return subprocess.call(cmd)

    # Add --only-context-length support via env for child (patch child args).
    # Children get a single physical GPU each.
    t0 = time.time()
    shard_specs = [
        (ids[0], "8000", f"{args.out_tag}_c8k"),
        (ids[1], "16000", f"{args.out_tag}_c16k"),
    ]
    procs = []
    for gpu, ctx, tag in shard_specs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PRIORITYKV_ONLY_CONTEXT_LENGTH"] = ctx
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_d4_latency.py"),
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
            "--m3-gate",
            "--e2e-gate-mult",
            str(args.e2e_gate_mult),
            "--ttft-gate-mult",
            str(args.ttft_gate_mult),
            "--tpot-gate-mult",
            str(args.tpot_gate_mult),
            "--pack-ms-max",
            str(args.pack_ms_max),
            "--cold-ms-max",
            str(args.cold_ms_max),
            "--pack-ms-max-16k",
            str(args.pack_ms_max_16k),
            "--cold-ms-max-16k",
            str(args.cold_ms_max_16k),
        ]
        log_path = out_dir / f"dual_{tag}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[dual] start ctx={ctx} gpu={gpu} tag={tag}", flush=True)
        procs.append(
            (
                tag,
                subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ROOT)),
                log_f,
                log_path,
            )
        )

    codes = []
    for tag, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        codes.append(rc)
        print(f"[dual] done tag={tag} exit={rc} log={log_path}", flush=True)

    shard_jsons = []
    for _, ctx, tag in shard_specs:
        p = out_dir / f"d4_latency_{tag}.json"
        if not p.exists():
            result = {
                "decision": "D4_M3_FAIL",
                "pass": False,
                "error": f"missing shard {p}",
                "exit_codes": codes,
            }
            out_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            print(f"out={out_path}")
            return 1
        shard_jsons.append(json.loads(p.read_text(encoding="utf-8")))

    merged = _merge(shard_jsons[0], shard_jsons[1])
    merged["tag"] = args.out_tag
    merged["seconds"] = round(time.time() - t0, 3)
    merged["exit_codes"] = codes
    out_path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": merged["decision"],
                "pass": merged["pass"],
                "m3": merged["m3"],
                "by_context": {
                    ctx: {
                        arm: {
                            "e2e_ttft_ms_mean": v.get("e2e_ttft_ms_mean"),
                            "pack_ms_mean": v.get("pack_ms_mean"),
                            "tpot_ms_mean": v.get("tpot_ms_mean"),
                            "score_mean": v.get("score_mean"),
                        }
                        for arm, v in arms.items()
                    }
                    for ctx, arms in (merged.get("by_context") or {}).items()
                },
                "seconds": merged["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if merged["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
