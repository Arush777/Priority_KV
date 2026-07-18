#!/usr/bin/env python3
"""Publish-track guardrails: FullKV vs structure-mixed @ int4_frac=0.75.

Extends W4 local RULER/SCBench-style probes; optional MATH-500 subsample.
Gate: mean |Δ| < 1pt on gate tasks (ruler_vt, scbench_choice, math if present).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: F401

from prioritykv.fullkv_compare import PromptRow, resolve_model_path
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig
from prioritykv.mixed_kv_run import run_transformers_mixed_kv

# Reuse task builders from W4 harness.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_guardrails", ROOT / "scripts" / "run_guardrails.py"
)
_gw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gw)


def _math_tasks(n: int = 50) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        print(f"[guardrails_pub] MATH skip: datasets import failed ({exc})", flush=True)
        return []
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as exc:  # noqa: BLE001
        print(f"[guardrails_pub] MATH skip: load failed ({exc})", flush=True)
        return []
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        prob = str(row.get("problem") or row.get("question") or "")
        ans = str(row.get("answer") or row.get("solution") or "")
        # Prefer boxed final if present
        gold = ans.strip().split("\n")[-1][:80]
        if "\\boxed{" in ans:
            try:
                gold = ans.split("\\boxed{")[-1].split("}")[0]
            except Exception:
                pass
        out.append(
            {
                "id": f"math500_{i}",
                "messages": [
                    {
                        "role": "user",
                        "content": prob
                        + "\n\nAnswer with the final numeric/expression only.",
                    }
                ],
                "gold": gold,
                "task": "math500",
            }
        )
    return out


def _score(text: str, gold: str) -> float:
    t = (text or "").strip().lower()
    g = (gold or "").strip().lower()
    if not g:
        return 0.0
    return 1.0 if g in t else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--math-n", type=int, default=50)
    ap.add_argument("--out-tag", default="pub_r1")
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "guardrails"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"guardrails_pub_{args.out_tag}.json"

    cfg = {
        "model": {
            "local_dirname": "Qwen3-8B",
            "hub_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        }
    }
    model_path = resolve_model_path(cfg)
    plan_cfg = MixedPlanConfig(
        int4_frac=0.75,
        sink_tokens=16,
        recent_window=128,
        risk_fit_path=str(ROOT / "configs" / "linear_risk_fit.json"),
    )
    int4_cfg = Int4KvConfig(nbits=4, group_size=32)

    tasks = _gw._build_tasks()
    math_items = _math_tasks(args.math_n)
    if math_items:
        tasks["math500"] = math_items

    results = {
        "manifest_id": "guardrails_pub",
        "rev": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "int4_frac": 0.75,
        "tasks": {},
        "rows": [],
        "math_n": len(math_items),
    }

    t0 = time.time()
    for name, items in tasks.items():
        prompts = [PromptRow(id=it["id"], messages=it["messages"]) for it in items]
        print(f"[guardrails_pub] task={name} n={len(items)} full…", flush=True)
        full_outs = run_transformers_mixed_kv(
            model_path,
            prompts,
            args.max_new_tokens,
            policy="full",
            plan_cfg=plan_cfg,
            int4_cfg=int4_cfg,
            storage="packed",
            attn_backend="flashinfer",
            fi_parity_every=0,
            fi_require_pass=False,
            max_model_len=8192,
        )
        print(f"[guardrails_pub] task={name} structure…", flush=True)
        struct_outs = run_transformers_mixed_kv(
            model_path,
            prompts,
            args.max_new_tokens,
            policy="structure",
            plan_cfg=plan_cfg,
            int4_cfg=int4_cfg,
            storage="packed",
            attn_backend="flashinfer",
            fi_parity_every=0,
            fi_require_pass=False,
            max_model_len=8192,
        )
        sf_list, sp_list = [], []
        for it, (ft, _, _), (st, _, _) in zip(items, full_outs, struct_outs, strict=True):
            sf = _score(ft, it["gold"])
            sp = _score(st, it["gold"])
            sf_list.append(sf)
            sp_list.append(sp)
            results["rows"].append(
                {
                    "id": it["id"],
                    "task": name,
                    "fullkv_score": sf,
                    "structure_score": sp,
                    "delta": sp - sf,
                }
            )
        mf = sum(sf_list) / len(sf_list)
        mp = sum(sp_list) / len(sp_list)
        results["tasks"][name] = {
            "n": len(items),
            "fullkv_mean": mf,
            "structure_mean": mp,
            "delta": mp - mf,
        }

    gate_tasks = ["ruler_vt", "scbench_choice"]
    if "math500" in results["tasks"]:
        gate_tasks.append("math500")
    max_abs = max(abs(float(results["tasks"][t]["delta"])) for t in gate_tasks)
    results["gate_tasks"] = gate_tasks
    results["max_abs_delta"] = max_abs
    results["pass"] = bool(max_abs <= args.threshold)
    results["decision"] = "GUARDRAILS_PUB_PASS" if results["pass"] else "GUARDRAILS_PUB_FAIL"
    results["seconds"] = round(time.time() - t0, 3)
    results["note"] = (
        "FullKV vs structure-mixed @0.75 on local RULER/SCBench-style probes"
        + (" + MATH-500 subsample" if math_items else " (MATH skipped)")
    )
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"decision={results['decision']} max_abs_delta={max_abs:.4f} "
        f"pass={results['pass']} out={out_path}"
    )
    print(f"out={out_path}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
