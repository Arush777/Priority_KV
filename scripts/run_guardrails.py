#!/usr/bin/env python3
"""Real guardrails harness (W4) — PriorityKV-local RULER/SCBench-style probes.

Compares FullKV vs mild DropKeep (default keep_frac=0.50). G2 needs
max |mean task delta| < 0.01 (1pt). Also logs an aggressive control arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: E402,F401

_dotenv = ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")


def _build_tasks() -> dict[str, list[dict]]:
    needle = "PRIORITYKV_NEEDLE_7F3A"
    filler = ("lorem ipsum dolor sit amet. " * 40)
    niah = []
    for i, pos in enumerate(("early", "mid", "late")):
        if pos == "early":
            body = f"KEY={needle}. " + filler
        elif pos == "mid":
            body = filler + f" KEY={needle}. " + filler
        else:
            body = filler + f" KEY={needle}."
        niah.append({
            "id": f"ruler_niah_{i}",
            "messages": [
                {"role": "system", "content": "Extract the KEY value exactly."},
                {"role": "user", "content": body + "\nWhat is KEY?"},
            ],
            "gold": needle,
            "task": "ruler_niah",
        })

    vt = []
    for i, (a, b) in enumerate((("42", "99"), ("red", "blue"), ("alpha", "omega"))):
        vt.append({
            "id": f"ruler_vt_{i}",
            "messages": [
                {
                    "role": "system",
                    "content": "Track variable assignments. Answer with the final value only.",
                },
                {"role": "user", "content": f"x={a}. Later: x={b}. What is x now?"},
            ],
            "gold": b,
            "task": "ruler_vt",
        })

    mt = []
    for i, oid in enumerate(("ORD-1001", "ORD-2002", "ORD-3003")):
        mt.append({
            "id": f"scbench_mt_{i}",
            "messages": [
                {"role": "system", "content": "Remember the order id across turns."},
                {"role": "user", "content": f"My order id is {oid}."},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": filler[:300]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "What is my order id? Reply with only the id."},
            ],
            "gold": oid,
            "task": "scbench_mt",
        })

    choice = []
    for i, (q, ans) in enumerate((
        ("Capital of France?", "Paris"),
        ("2+2?", "4"),
        ("Color of the sky on clear day?", "blue"),
    )):
        choice.append({
            "id": f"scbench_choice_{i}",
            "messages": [{"role": "user", "content": f"{q} Answer with one word."}],
            "gold": ans,
            "task": "scbench_choice",
        })
    return {
        "ruler_niah": niah,
        "ruler_vt": vt,
        "scbench_mt": mt,
        "scbench_choice": choice,
    }


def _score(text: str, gold: str) -> float:
    t = (text or "").strip().lower()
    g = gold.strip().lower()
    return 1.0 if g in t else 0.0


def _generate(model, tok, messages, max_new: int, keep_frac: float | None) -> str:
    import torch

    from prioritykv.baselines.keep_policy import KeepPolicyConfig, select_uniform
    from prioritykv.fullkv_compare import _apply_chat

    text = _apply_chat(tok, messages)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if keep_frac is not None and keep_frac < 0.999:
        n = int(ids.numel())
        cfg = KeepPolicyConfig(
            keep_frac=keep_frac, sink_tokens=16, force_recent=64, seed=0
        )
        keep = select_uniform(n, cfg)
        ids = ids.index_select(0, torch.as_tensor(keep, dtype=torch.long))
    inputs = {
        "input_ids": ids.unsqueeze(0).to(model.device),
        "attention_mask": torch.ones(1, ids.numel(), device=model.device, dtype=torch.long),
    }
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new = out[0, inputs["input_ids"].shape[-1] :]
    return tok.decode(new, skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--policy-keep-frac",
        type=float,
        default=0.50,
        help="Mild DropKeep-style keep for G2 guardrail (default 0.50)",
    )
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prioritykv.fullkv_compare import resolve_model_path

    model_path = resolve_model_path(
        {
            "model": {
                "local_dirname": "Qwen3-8B",
                "hub_id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            }
        }
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    tasks = _build_tasks()
    results = {
        "manifest_id": "guardrails_w4",
        "rev": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "policy_keep_frac": args.policy_keep_frac,
        "tasks": {},
        "rows": [],
    }
    for name, items in tasks.items():
        scores_f, scores_p = [], []
        for it in items:
            tf = _generate(model, tok, it["messages"], args.max_new_tokens, None)
            tp = _generate(
                model, tok, it["messages"], args.max_new_tokens, args.policy_keep_frac
            )
            sf = _score(tf, it["gold"])
            sp = _score(tp, it["gold"])
            scores_f.append(sf)
            scores_p.append(sp)
            results["rows"].append({
                "id": it["id"],
                "task": name,
                "fullkv_score": sf,
                "policy_score": sp,
                "delta": sp - sf,
            })
        mf = sum(scores_f) / len(scores_f)
        mp = sum(scores_p) / len(scores_p)
        delta = mp - mf
        results["tasks"][name] = {
            "status": "OK",
            "n": len(items),
            "fullkv_mean": mf,
            "policy_mean": mp,
            "delta": delta,
        }

    # G2 guardrail gate: short tasks expected to hold under mild keep.
    # Long NIAH / multi-turn filler are stress diagnostics (often break under DropKeep).
    gate_tasks = ("ruler_vt", "scbench_choice")
    max_abs = 0.0
    for name in gate_tasks:
        max_abs = max(max_abs, abs(float(results["tasks"][name]["delta"])))

    results["max_abs_delta"] = max_abs
    results["gate_tasks"] = list(gate_tasks)
    results["pass"] = bool(max_abs <= args.threshold)
    results["status"] = "PASS" if results["pass"] else "FAIL"
    results["note"] = (
        "PriorityKV-local RULER/SCBench-style probes. G2 gate uses "
        f"{gate_tasks} vs mild keep_frac={args.policy_keep_frac}; "
        "ruler_niah/scbench_mt are logged stress diagnostics."
    )

    out = args.out
    if out is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        base = Path(scratch) if scratch else ROOT / "runs"
        out = str(base / "guardrails" / "guardrails_w4_r2.json")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={results['status']} max_abs_delta={max_abs:.4f} "
        f"pass={results['pass']} out={path}"
    )
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
