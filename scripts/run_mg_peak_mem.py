#!/usr/bin/env python3
"""Middle-ground peak CUDA memory + packed payload bytes microbench.

Reports, per arm (FullKV SDPA / uniform FI / structure FI):
  - payload_bytes (packed K/V storage; 0 for FullKV)
  - peak_allocated_bytes during decode (torch.cuda.max_memory_allocated)
  - peak_reserved_bytes
  - compression vs modeled FullKV BF16 KV bytes

Honest caveat (M3c): FI cold scratch expands INT4 pages to BF16 for attend —
peak_allocated can stay near FullKV even when payload_bytes ≪ FullKV.
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
from prioritykv.baselines.buried_state import relocate_state_to_middle
from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.bench_pilot import materialize_examples
from prioritykv.fi_mixed_decode import eager_prepare_decode
from prioritykv.fullkv_compare import PromptRow, _apply_chat, resolve_model_path
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig, plan_int4_mask
from prioritykv.page_roles import PageRole
from prioritykv.qwen3_fi_shim import (
    FiSeqLenCache,
    fi_shim_context,
    pack_prefill_to_fi_state,
)
from prioritykv.stress_pilot import select_stress_rows


def _cuda_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _peak_snapshot() -> dict[str, int]:
    import torch

    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _arm_mem_reset() -> None:
    """Reset peak counters and drop unused cached blocks between arms."""
    import torch

    _cuda_sync()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _cuda_sync()


def _run_fullkv(model, tok, ids, *, max_new_tokens: int) -> dict[str, Any]:
    import torch

    n = int(ids.numel())
    cache_n = n - 1
    device = model.device
    _arm_mem_reset()
    with torch.no_grad():
        pre = model(
            input_ids=ids[:cache_n].unsqueeze(0),
            attention_mask=torch.ones(1, cache_n, dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        past = pre.past_key_values
        # Match FI arm: measure decode peak after prefill is resident (Opus note).
        _arm_mem_reset()
        attn = torch.ones(1, n, dtype=torch.long, device=device)
        replay = model(
            input_ids=ids[-1:].view(1, 1),
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = replay.past_key_values
        next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
        cur = torch.tensor([[next_id]], device=device)
        for _ in range(max_new_tokens - 1):
            attn = torch.cat(
                [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1
            )
            step = model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = step.past_key_values
            nid = int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
            cur = torch.tensor([[nid]], device=device)
    _cuda_sync()
    peak = _peak_snapshot()
    return {
        "arm": "fullkv_sdpa",
        "payload_bytes": 0,
        "fullkv_bf16_bytes": None,
        "compression_ratio": 1.0,
        **peak,
    }


def _run_fi(
    model,
    tok,
    ids,
    *,
    roles,
    plan_cfg: MixedPlanConfig,
    policy: str,
    int4_cfg: Int4KvConfig,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    n = int(ids.numel())
    cache_n = n - 1
    device = model.device
    role_list = list(roles)
    if len(role_list) != cache_n:
        if len(role_list) > cache_n:
            role_list = role_list[:cache_n]
        else:
            role_list = role_list + [PageRole.RECENT] * (cache_n - len(role_list))
    mask = plan_int4_mask(role_list, plan_cfg, policy=policy)

    # Prefill+pack may briefly double-resident HF past; we measure decode peak
    # after pack/cold and releasing HF past (Codex P1).
    with torch.no_grad():
        pre = model(
            input_ids=ids[:cache_n].unsqueeze(0),
            attention_mask=torch.ones(1, cache_n, dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        past = pre.past_key_values
        packed, state = pack_prefill_to_fi_state(
            past,
            role_list,
            mask,
            device=device,
            dtype=torch.bfloat16,
            int4_cfg=int4_cfg,
            decode_tail_cap=max(max_new_tokens + 8, 64),
        )
        del past, pre
        eager_prepare_decode(state)
        stub = FiSeqLenCache(state)
        _arm_mem_reset()
        attn = torch.ones(1, n, dtype=torch.long, device=device)
        with fi_shim_context(state) as ctx:
            replay = model(
                input_ids=ids[-1:].view(1, 1),
                attention_mask=attn,
                past_key_values=stub,
                use_cache=True,
                return_dict=True,
            )
            next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
            cur = torch.tensor([[next_id]], device=device)
            for _ in range(max_new_tokens - 1):
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)],
                    dim=1,
                )
                step = model(
                    input_ids=cur,
                    attention_mask=attn,
                    past_key_values=stub,
                    use_cache=True,
                    return_dict=True,
                )
                nid = int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
                cur = torch.tensor([[nid]], device=device)
            used_mat = bool(ctx.used_materialize)
        state.assert_no_materialize_path(used_mat)
    _cuda_sync()
    peak = _peak_snapshot()
    return {
        "arm": f"mixed_{policy}_fi_shim",
        "payload_bytes": int(packed.payload_bytes()),
        "fullkv_bf16_bytes": int(packed.fullkv_bf16_bytes()),
        "compression_ratio": round(packed.compression_ratio(), 6),
        "int4_tokens": int(mask.sum()),
        "used_materialize_hf_past": used_mat,
        **peak,
    }


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mg_peak_mem.yaml"))
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "mg_peak_mem"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mg_peak_mem_{args.out_tag}.json"

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
    bench = json.loads((ROOT / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=ROOT / "data" / "prioritybench")
    use_middle = bool(cfg.get("relocate_middle", False))
    middle_pos = float(cfg.get("relocate_position", 0.5))

    model_path = resolve_model_path(cfg)
    mcfg = cfg.get("mixed", {})
    risk_path = mcfg.get("risk_fit_path")
    if risk_path:
        rp = Path(str(risk_path))
        risk_path = str(rp if rp.is_absolute() else ROOT / rp)
    plan_cfg = MixedPlanConfig(
        int4_frac=float(mcfg.get("int4_frac", 0.75)),
        sink_tokens=int(mcfg.get("sink_tokens", 16)),
        recent_window=int(mcfg.get("recent_window", 128)),
        risk_fit_path=risk_path,
    )
    int4_cfg = Int4KvConfig(
        nbits=int(mcfg.get("nbits", 4)),
        group_size=int(mcfg.get("group_size", 32)),
    )
    policies = [p for p in mcfg.get("policies", ["full", "uniform", "structure"]) if p != "full"]
    max_new = int(args.max_new_tokens)

    print(f"[peak_mem] loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    chat_kwargs = dict(qwen3_chat_template_kwargs())

    t0 = time.time()
    detail: list[dict[str, Any]] = []
    for ex in examples:
        msgs = list(ex.messages)
        if use_middle:
            msgs = relocate_state_to_middle(
                msgs, position=middle_pos, seed=hash(ex.example_id) % 10_000
            )
        prompt = PromptRow(id=ex.example_id, messages=msgs)
        text = _apply_chat(tok, prompt.messages)
        ids = tok(text, return_tensors="pt")["input_ids"][0].to(model.device)
        roles = assign_token_roles(tok, prompt.messages, chat_kwargs=chat_kwargs)

        row_base = {
            "example_id": ex.example_id,
            "category": ex.category.value,
            "context_length": int(ex.context_length),
            "prompt_tokens": int(ids.numel()),
        }
        full = _run_fullkv(model, tok, ids, max_new_tokens=max_new)
        detail.append({**row_base, **full})
        print(
            f"[peak_mem] {ex.example_id} full peak={full['peak_allocated_bytes']/1e9:.2f}GB",
            flush=True,
        )
        for policy in policies:
            fi = _run_fi(
                model,
                tok,
                ids,
                roles=roles,
                plan_cfg=plan_cfg,
                policy=policy,
                int4_cfg=int4_cfg,
                max_new_tokens=max_new,
            )
            detail.append({**row_base, **fi})
            print(
                f"[peak_mem] {ex.example_id} {policy} "
                f"peak={fi['peak_allocated_bytes']/1e9:.2f}GB "
                f"payload={fi['payload_bytes']/1e6:.1f}MB "
                f"comp={fi['compression_ratio']}",
                flush=True,
            )

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in detail:
        by_arm.setdefault(r["arm"], []).append(r)

    arms = {}
    for arm, rs in by_arm.items():
        payload_m = _mean([r.get("payload_bytes") for r in rs])
        full_m = _mean([r.get("fullkv_bf16_bytes") for r in rs])
        measured_ratio = (
            None
            if payload_m is None or full_m is None or float(full_m) <= 0
            else float(payload_m) / float(full_m)
        )
        arms[arm] = {
            "n": len(rs),
            "peak_allocated_bytes_mean": _mean([r.get("peak_allocated_bytes") for r in rs]),
            "peak_reserved_bytes_mean": _mean([r.get("peak_reserved_bytes") for r in rs]),
            "payload_bytes_mean": payload_m,
            "fullkv_bf16_bytes_mean": full_m,
            "payload_ratio_measured_mean": measured_ratio,
            "compression_ratio_modeled_mean": _mean(
                [r.get("compression_ratio") for r in rs]
            ),
        }

    full_peak = arms.get("fullkv_sdpa", {}).get("peak_allocated_bytes_mean")
    struct = arms.get("mixed_structure_fi_shim", {})
    # Gate on measured payload (uint8 codes), not idealized bit-pack model (Codex P1).
    measured = struct.get("payload_ratio_measured_mean")
    payload_ok = measured is not None and float(measured) < 0.85
    no_mat = all(
        not r.get("used_materialize_hf_past")
        for r in detail
        if r["arm"] != "fullkv_sdpa"
    )
    decision = "MG_PEAK_MEM_PASS" if payload_ok and no_mat else "MG_PEAK_MEM_GATE_FAIL"
    result = {
        "job": "mg_peak_mem",
        "tag": args.out_tag,
        "decision": decision,
        "pass": decision == "MG_PEAK_MEM_PASS",
        "n_examples": len(examples),
        "arms": arms,
        "fullkv_peak_allocated_bytes_mean": full_peak,
        "structure_vs_fullkv_peak_ratio": (
            None
            if full_peak is None or not struct.get("peak_allocated_bytes_mean")
            else float(struct["peak_allocated_bytes_mean"]) / float(full_peak)
        ),
        "caveat": (
            "FI cold scratch expands INT4→BF16 for attend; peak_allocated can stay "
            "near FullKV while payload_bytes ≪ FullKV. Report both. "
            "payload_ratio_measured uses actual uint8+FP32 payloads; "
            "compression_ratio_modeled uses idealized 0.5B/elem INT4 model."
        ),
        "seconds": round(time.time() - t0, 3),
        "device": str(model.device),
        "rows": detail,
    }
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "pass": result["pass"],
                "arms": {
                    k: {
                        "peak_GB": (v["peak_allocated_bytes_mean"] or 0) / 1e9,
                        "payload_MB": (v["payload_bytes_mean"] or 0) / 1e6,
                        "payload_ratio_measured": v.get("payload_ratio_measured_mean"),
                        "compression_modeled": v.get("compression_ratio_modeled_mean"),
                    }
                    for k, v in arms.items()
                },
                "structure_vs_fullkv_peak_ratio": result["structure_vs_fullkv_peak_ratio"],
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
