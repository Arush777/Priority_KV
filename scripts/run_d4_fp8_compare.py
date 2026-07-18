#!/usr/bin/env python3
"""D4b: FullKV SDPA vs vLLM FP8 vs structure-FI @ int4_frac=0.75.

Paper systems table: quality + e2e/TPOT where available + modeled byte ratio.
Gate = completeness (all arms scored), not forced FP8 latency win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: F401

from prioritybench.pins import qwen3_chat_template_kwargs
from prioritybench.scoring import score_example
from prioritykv.baselines.buried_state import relocate_state_to_middle
from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.bench_pilot import materialize_examples
from prioritykv.byte_model import fullkv_bf16_bytes, realized_bytes
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, _apply_chat, resolve_model_path
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig
from prioritykv.stress_pilot import select_stress_rows

# Load D4 helpers without requiring scripts/ on PYTHONPATH as a package.
import importlib.util

_d4_path = ROOT / "scripts" / "run_d4_latency.py"
_spec = importlib.util.spec_from_file_location("run_d4_latency", _d4_path)
_d4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_d4)
_arm_summary = _d4._arm_summary
_mean = _d4._mean
_median = _d4._median
_text_for_score = _d4._text_for_score
_timed_decode_fi = _d4._timed_decode_fi
_timed_decode_fullkv = _d4._timed_decode_fullkv


def _modeled_fp8_bytes(n_tokens: int) -> int:
    """Approx FP8 KV = half of BF16 realized (no scales)."""
    return max(1, fullkv_bf16_bytes(n_tokens) // 2)


def _modeled_mixed_bytes(n_tokens: int, int4_frac: float) -> int:
    n_int4 = int(round(n_tokens * int4_frac))
    n_bf16 = n_tokens - n_int4
    return realized_bytes(
        num_bf16_tokens=n_bf16,
        num_int4_tokens=n_int4,
        num_kv_heads=8,
        head_dim=128,
        page_tokens=16,
        num_layers=36,
    )


def main() -> int:
    import yaml

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

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        result = {"decision": "SKIP_NO_TORCH", "error": str(exc), "pass": None}
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 0

    if not torch.cuda.is_available():
        result = {"decision": "SKIP_NO_CUDA", "pass": None}
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 0

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    only_ctx = os.environ.get("PRIORITYKV_ONLY_CONTEXT_LENGTH")
    if only_ctx:
        cfg = dict(cfg)
        sel = dict(cfg.get("selection") or {})
        sel["context_lengths"] = [int(only_ctx)]
        cfg["selection"] = sel
        print(f"[fp8cmp] ONLY_CONTEXT_LENGTH={only_ctx}", flush=True)

    bench = json.loads((ROOT / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=ROOT / "data" / "prioritybench")
    prompts: list[PromptRow] = []
    for ex in examples:
        msgs = list(ex.messages)
        if cfg.get("relocate_middle"):
            msgs = relocate_state_to_middle(
                msgs,
                position=float(cfg.get("relocate_position", 0.5)),
                seed=hash(ex.example_id) % 10_000,
            )
        prompts.append(PromptRow(id=ex.example_id, messages=msgs))

    model_path = resolve_model_path(cfg)
    mcfg = cfg.get("mixed", {})
    risk = mcfg.get("risk_fit_path")
    plan_cfg = MixedPlanConfig(
        int4_frac=float(mcfg.get("int4_frac", 0.75)),
        sink_tokens=int(mcfg.get("sink_tokens", 16)),
        recent_window=int(mcfg.get("recent_window", 128)),
        risk_fit_path=str(ROOT / risk) if risk else None,
    )
    int4_cfg = Int4KvConfig(
        nbits=int(mcfg.get("nbits", 4)),
        group_size=int(mcfg.get("group_size", 32)),
    )
    max_new = int(args.max_new_tokens or cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    int4_frac = float(mcfg.get("int4_frac", 0.75))

    print(f"[fp8cmp] n={len(prompts)} loading HF {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_kwargs = dict(qwen3_chat_template_kwargs())
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    def _ids_for(prompt: PromptRow):
        text = _apply_chat(tok, prompt.messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        budget = int(vcfg["max_model_len"]) - max_new - 8
        if ids.numel() > budget:
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])
        return ids.to(model.device)

    if prompts and args.warmup > 0:
        w_ids = _ids_for(prompts[0])
        w_roles = assign_token_roles(tok, prompts[0].messages, chat_kwargs=chat_kwargs)
        _timed_decode_fullkv(model, tok, w_ids, max_new_tokens=min(8, max_new))
        _timed_decode_fi(
            model,
            tok,
            w_ids,
            roles=w_roles,
            plan_cfg=plan_cfg,
            policy="structure",
            int4_cfg=int4_cfg,
            max_new_tokens=min(8, max_new),
        )

    rows_out: list[dict[str, Any]] = []
    repeats = max(1, int(args.repeats))
    timing_keys = (
        "prefill_ms",
        "pack_ms",
        "cold_scratch_ms",
        "decode_ttft_ms",
        "e2e_ttft_ms",
        "tpot_ms",
        "tokens_per_s",
    )

    t_hf0 = time.time()
    for prompt, ex in zip(prompts, examples, strict=True):
        ids = _ids_for(prompt)
        roles = assign_token_roles(tok, prompt.messages, chat_kwargs=chat_kwargs)
        ntok = int(ids.numel())
        ctx = int(ex.context_length)

        def _run(kind: str, policy: str | None = None) -> dict[str, Any]:
            reps = []
            for _ in range(repeats):
                if kind == "fullkv":
                    reps.append(
                        _timed_decode_fullkv(model, tok, ids, max_new_tokens=max_new)
                    )
                else:
                    reps.append(
                        _timed_decode_fi(
                            model,
                            tok,
                            ids,
                            roles=roles,
                            plan_cfg=plan_cfg,
                            policy=policy or "structure",
                            int4_cfg=int4_cfg,
                            max_new_tokens=max_new,
                        )
                    )
            out = dict(reps[-1])
            for k in timing_keys:
                out[k] = _median([r.get(k) for r in reps])
            out["example_id"] = prompt.id
            out["prompt_tokens"] = ntok
            out["context_length"] = ctx
            out["category"] = ex.category.value
            out["score"] = float(
                score_example(
                    ex,
                    _text_for_score(tok, list(out.get("token_ids") or []), out["text"]),
                )
            )
            out["fullkv_bf16_bytes_modeled"] = fullkv_bf16_bytes(ntok)
            out["mixed_bytes_modeled"] = _modeled_mixed_bytes(ntok, int4_frac)
            out["fp8_bytes_modeled"] = _modeled_fp8_bytes(ntok)
            return out

        full = _run("fullkv")
        struct = _run("fi", "structure")
        rows_out.extend([full, struct])
        print(
            f"[fp8cmp/hf] {prompt.id} full_score={full['score']:.2f} "
            f"struct_score={struct['score']:.2f} e2e={full.get('e2e_ttft_ms'):.0f}/"
            f"{struct.get('e2e_ttft_ms'):.0f}ms",
            flush=True,
        )
    hf_seconds = time.time() - t_hf0

    # Free HF model before vLLM (same GPU).
    del model
    torch.cuda.empty_cache()

    print("[fp8cmp] running vLLM FP8 quality+wall…", flush=True)
    t_fp0 = time.time()
    try:
        fp8_outs = _run_vllm_mode(
            model_path,
            prompts,
            max_new_tokens=max_new,
            kv_cache_dtype=str(cfg.get("fp8", {}).get("kv_cache_dtype", "fp8")),
            calculate_kv_scales=bool(cfg.get("fp8", {}).get("calculate_kv_scales", True)),
            tp=int(vcfg["tensor_parallel_size"]),
            gpu_mem=float(vcfg["gpu_memory_utilization"]),
            max_model_len=int(vcfg["max_model_len"]),
        )
        fp8_ok = True
        fp8_err = None
    except Exception as exc:  # noqa: BLE001
        fp8_outs = [("", [])] * len(prompts)
        fp8_ok = False
        fp8_err = str(exc)
        print(f"[fp8cmp] FP8 FAILED: {exc}", flush=True)
    fp8_seconds = time.time() - t_fp0
    per_ex_fp8_ms = (fp8_seconds / max(len(prompts), 1)) * 1000.0

    for prompt, ex, (txt, tids) in zip(prompts, examples, fp8_outs, strict=True):
        ntok = len(
            tok(
                _apply_chat(tok, prompt.messages),
                add_special_tokens=False,
            )["input_ids"]
        )
        score = float(score_example(ex, _text_for_score(tok, list(tids), txt))) if fp8_ok else None
        rows_out.append(
            {
                "arm": "vllm_fp8",
                "example_id": prompt.id,
                "context_length": int(ex.context_length),
                "category": ex.category.value,
                "prompt_tokens": ntok,
                "score": score,
                "text": txt,
                "e2e_ttft_ms": per_ex_fp8_ms if fp8_ok else None,
                "tpot_ms": None,
                "pack_ms": 0.0,
                "cold_scratch_ms": 0.0,
                "decode_ttft_ms": None,
                "note": "e2e_ttft_ms = batch_wall/n (vLLM); not phase-matched to HF",
                "fullkv_bf16_bytes_modeled": fullkv_bf16_bytes(ntok),
                "fp8_bytes_modeled": _modeled_fp8_bytes(ntok),
                "mixed_bytes_modeled": _modeled_mixed_bytes(ntok, int4_frac),
                "byte_ratio_vs_fullkv_modeled": 0.5,
            }
        )

    by_arm: dict[str, list] = {}
    for r in rows_out:
        by_arm.setdefault(r["arm"], []).append(r)
    arms = {a: _arm_summary(rs) for a, rs in by_arm.items()}
    # Attach byte means
    for a, rs in by_arm.items():
        arms[a]["fp8_bytes_modeled_mean"] = _mean([r.get("fp8_bytes_modeled") for r in rs])
        arms[a]["mixed_bytes_modeled_mean"] = _mean(
            [r.get("mixed_bytes_modeled") for r in rs]
        )
        arms[a]["fullkv_bf16_bytes_modeled_mean"] = _mean(
            [r.get("fullkv_bf16_bytes_modeled") for r in rs]
        )

    by_ctx: dict[str, dict] = {}
    for r in rows_out:
        ctx = str(int(r.get("context_length") or 0))
        by_ctx.setdefault(ctx, {}).setdefault(r["arm"], []).append(r)
    summary_by_ctx = {
        ctx: {arm: _arm_summary(rs) for arm, rs in arms.items()}
        for ctx, arms in by_ctx.items()
    }

    n_ok = all(
        arms.get(a, {}).get("n", 0) == len(examples)
        for a in ("fullkv_sdpa", "mixed_structure_fi_shim", "vllm_fp8")
    )
    decision = "D4_FP8_COMPARE_PASS" if (fp8_ok and n_ok) else "D4_FP8_COMPARE_PARTIAL"
    result = {
        "job": "d4_fp8_compare",
        "tag": args.out_tag,
        "decision": decision,
        "pass": decision == "D4_FP8_COMPARE_PASS",
        "n_examples": len(examples),
        "fp8_ok": fp8_ok,
        "fp8_error": fp8_err,
        "int4_frac": int4_frac,
        "arms": arms,
        "by_context": summary_by_ctx,
        "hf_seconds": round(hf_seconds, 3),
        "fp8_seconds": round(fp8_seconds, 3),
        "note": (
            "FP8 e2e is batch-amortized wall/n; HF arms use D4 phase timing. "
            "Byte ratios are modeled. Reliability-at-parity reframe allowed if "
            "structure does not beat FP8 on latency."
        ),
        "rows": rows_out,
    }
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "pass": result["pass"],
                "arms": {
                    k: {
                        "score_mean": v.get("score_mean"),
                        "e2e_ttft_ms_mean": v.get("e2e_ttft_ms_mean"),
                        "tpot_ms_mean": v.get("tpot_ms_mean"),
                    }
                    for k, v in arms.items()
                },
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
