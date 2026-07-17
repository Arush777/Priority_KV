#!/usr/bin/env python3
"""Stage-1b smoke: Qwen3 FI-shim greedy vs materialize→SDPA.

Acceptance (Fable):
  - FI path never calls materialize_hf_past
  - N greedy token IDs match materialize→SDPA on a tiny prompt
  - peak_mem logged (soft)

H200: jobs/pending/w8_fi_greedy_smoke_r1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: F401

from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.fullkv_compare import _apply_chat
from prioritykv.mixed_kv import MixedPlanConfig
from prioritykv.qwen3_fi_shim import greedy_fi_decode, greedy_materialize_baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--int4-frac", type=float, default=0.5)
    ap.add_argument("--prompt", default="Reply with exactly: OK")
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "fi_greedy_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get(
        "PRIORITYKV_MODEL",
        str(scratch / "models" / "Qwen3-8B"),
    )

    result: dict = {
        "job": "fi_greedy_smoke",
        "tag": args.out_tag,
        "model_path": model_path,
    }
    t0 = time.time()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        result.update({"decision": "SKIP_NO_TORCH", "error": str(exc), "pass": None})
        _write(out_dir, args.out_tag, result)
        return 0

    if not torch.cuda.is_available():
        result.update({"decision": "SKIP_NO_CUDA", "pass": None})
        _write(out_dir, args.out_tag, result)
        return 0

    from prioritykv.flashinfer_multicall import flashinfer_available

    if not flashinfer_available():
        result.update({"decision": "SKIP_NO_FLASHINFER", "pass": None})
        _write(out_dir, args.out_tag, result)
        return 0

    from prioritybench.pins import qwen3_chat_template_kwargs

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    chat_kwargs = dict(qwen3_chat_template_kwargs())
    messages = [{"role": "user", "content": args.prompt}]
    text = _apply_chat(tok, messages)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    roles = assign_token_roles(tok, messages, chat_kwargs=chat_kwargs)
    plan = MixedPlanConfig(
        int4_frac=float(args.int4_frac),
        sink_tokens=4,
        recent_window=16,
    )

    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    base = greedy_materialize_baseline(
        model,
        tok,
        ids,
        roles=roles,
        plan_cfg=plan,
        policy="structure",
        max_new_tokens=args.max_new_tokens,
    )
    peak_base = int(torch.cuda.max_memory_allocated())

    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    try:
        fi = greedy_fi_decode(
            model,
            tok,
            ids,
            roles=roles,
            plan_cfg=plan,
            policy="structure",
            max_new_tokens=args.max_new_tokens,
        )
        fi_err = None
    except Exception as exc:  # noqa: BLE001
        fi = {"token_ids": [], "used_materialize_hf_past": None, "error": str(exc)}
        fi_err = str(exc)
    peak_fi = int(torch.cuda.max_memory_allocated())

    base_ids = list(base["token_ids"])
    fi_ids = list(fi.get("token_ids") or [])
    exact = base_ids == fi_ids and len(base_ids) > 0
    first_ok = bool(base_ids) and bool(fi_ids) and base_ids[0] == fi_ids[0]
    no_mat = fi.get("used_materialize_hf_past") is False

    if fi_err:
        decision, ok = "FAIL_EXCEPTION", False
    elif not no_mat:
        decision, ok = "FAIL_MATERIALIZE", False
    elif exact:
        decision, ok = "GREEDY_PASS", True
    elif first_ok:
        decision, ok = "SOFT_FIRST_MATCH", False
    else:
        decision, ok = "FAIL_TOKEN_MISMATCH", False

    result.update(
        {
            "decision": decision,
            "pass": ok,
            "base_ids": base_ids,
            "fi_ids": fi_ids,
            "exact_match": exact,
            "first_token_match": first_ok,
            "used_materialize_hf_past": fi.get("used_materialize_hf_past"),
            "fi_meta": {k: v for k, v in fi.items() if k not in ("text",)},
            "base_text": base.get("text"),
            "fi_text": fi.get("text"),
            "error": fi_err,
            "cuda_peak_bytes_baseline": peak_base,
            "cuda_peak_bytes_fi": peak_fi,
            "prompt_tokens": int(ids.numel()),
            "int4_frac": args.int4_frac,
            "max_new_tokens": args.max_new_tokens,
            "seconds": round(time.time() - t0, 3),
            "device": torch.cuda.get_device_name(0),
        }
    )
    path = _write(out_dir, args.out_tag, result)
    print(
        json.dumps(
            {
                "decision": decision,
                "pass": ok,
                "exact_match": exact,
                "used_materialize_hf_past": fi.get("used_materialize_hf_past"),
                "base_ids": base_ids,
                "fi_ids": fi_ids,
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={path}")
    return 0 if ok else 1


def _write(out_dir: Path, tag: str, result: dict) -> Path:
    path = out_dir / f"fi_greedy_smoke_{tag}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
