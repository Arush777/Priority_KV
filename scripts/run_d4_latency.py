#!/usr/bin/env python3
"""D4 microbench: TTFT / TPOT / peak mem for FullKV vs mixed FI-shim.

H200: enqueue via jobs/pending. Prints out=… for worker result push.
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

from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.fullkv_compare import PromptRow, _apply_chat, resolve_model_path
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig
from prioritykv.page_roles import PageRole
from prioritykv.qwen3_fi_shim import (
    FiSeqLenCache,
    fi_shim_context,
    pack_prefill_to_fi_state,
)
from prioritykv.stress_pilot import select_stress_rows
from prioritybench.scoring import score_example
from prioritykv.baselines.buried_state import relocate_state_to_middle
from prioritykv.bench_pilot import materialize_examples


def _cuda_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_decode_fullkv(
    model,
    tok,
    ids,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Split-prefill FullKV SDPA; TTFT = first replay token, then TPOT."""
    import torch

    n = int(ids.numel())
    cache_n = n - 1
    device = model.device
    _cuda_sync()
    t_pre0 = time.perf_counter()
    with torch.no_grad():
        pre = model(
            input_ids=ids[:cache_n].unsqueeze(0),
            attention_mask=torch.ones(1, cache_n, dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        past = pre.past_key_values
        _cuda_sync()
        t_pre1 = time.perf_counter()

        attn = torch.ones(1, n, dtype=torch.long, device=device)
        t0 = time.perf_counter()
        replay = model(
            input_ids=ids[-1:].view(1, 1),
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        _cuda_sync()
        t1 = time.perf_counter()
        past = replay.past_key_values
        next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
        gen = [next_id]
        cur = torch.tensor([[next_id]], device=device)
        step_ms: list[float] = []
        for _ in range(max_new_tokens - 1):
            attn = torch.cat(
                [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1
            )
            ts = time.perf_counter()
            step = model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            _cuda_sync()
            te = time.perf_counter()
            step_ms.append((te - ts) * 1000.0)
            past = step.past_key_values
            nid = int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
            gen.append(nid)
            cur = torch.tensor([[nid]], device=device)
            if tok.eos_token_id is not None and nid == tok.eos_token_id:
                break
    ttft_ms = (t1 - t0) * 1000.0
    tpot_ms = (sum(step_ms) / len(step_ms)) if step_ms else None
    return {
        "arm": "fullkv_sdpa",
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prefill_ms": (t_pre1 - t_pre0) * 1000.0,
        "n_new": len(gen),
        "tokens_per_s": (1000.0 / tpot_ms) if tpot_ms else None,
        "token_ids": gen,
        "text": tok.decode(gen, skip_special_tokens=True),
        "used_materialize_hf_past": False,
        "attn_backend": "sdpa",
    }


def _timed_decode_fi(
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
    from prioritykv.mixed_kv import plan_int4_mask

    mask = plan_int4_mask(role_list, plan_cfg, policy=policy)
    _cuda_sync()
    t_pre0 = time.perf_counter()
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
        _cuda_sync()
        t_pre1 = time.perf_counter()

        stub = FiSeqLenCache(state)
        attn = torch.ones(1, n, dtype=torch.long, device=device)
        with fi_shim_context(state) as ctx:
            t0 = time.perf_counter()
            replay = model(
                input_ids=ids[-1:].view(1, 1),
                attention_mask=attn,
                past_key_values=stub,
                use_cache=True,
                return_dict=True,
            )
            _cuda_sync()
            t1 = time.perf_counter()
            next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
            gen = [next_id]
            cur = torch.tensor([[next_id]], device=device)
            step_ms: list[float] = []
            for _ in range(max_new_tokens - 1):
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)],
                    dim=1,
                )
                ts = time.perf_counter()
                step = model(
                    input_ids=cur,
                    attention_mask=attn,
                    past_key_values=stub,
                    use_cache=True,
                    return_dict=True,
                )
                _cuda_sync()
                te = time.perf_counter()
                step_ms.append((te - ts) * 1000.0)
                nid = int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
                if tok.eos_token_id is not None and nid == tok.eos_token_id:
                    break
            used_mat = bool(ctx.used_materialize)
        state.assert_no_materialize_path(used_mat)
    ttft_ms = (t1 - t0) * 1000.0
    tpot_ms = (sum(step_ms) / len(step_ms)) if step_ms else None
    return {
        "arm": f"mixed_{policy}_fi_shim",
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prefill_pack_ms": (t_pre1 - t_pre0) * 1000.0,
        "n_new": len(gen),
        "tokens_per_s": (1000.0 / tpot_ms) if tpot_ms else None,
        "token_ids": gen,
        "text": tok.decode(gen, skip_special_tokens=True),
        "used_materialize_hf_past": used_mat,
        "attn_backend": "flashinfer_fi_shim",
        "int4_tokens": int(mask.sum()),
        "payload_bytes": packed.payload_bytes(),
        "compression_ratio": round(packed.compression_ratio(), 6),
    }


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w9_mixed_fi_decode.yaml"))
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "d4_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d4_latency_{args.out_tag}.json"

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
    prompts: list[PromptRow] = []
    for ex in examples:
        msgs = list(ex.messages)
        if cfg.get("relocate_middle"):
            msgs = relocate_state_to_middle(
                msgs, position=float(cfg.get("relocate_position", 0.5)), seed=hash(ex.example_id) % 10_000
            )
        prompts.append(PromptRow(id=ex.example_id, messages=msgs))

    model_path = resolve_model_path(cfg)
    mcfg = cfg.get("mixed", {})
    plan_cfg = MixedPlanConfig(
        int4_frac=float(mcfg.get("int4_frac", 0.75)),
        sink_tokens=int(mcfg.get("sink_tokens", 16)),
        recent_window=int(mcfg.get("recent_window", 128)),
        risk_fit_path=str(ROOT / mcfg["risk_fit_path"])
        if mcfg.get("risk_fit_path")
        else None,
    )
    int4_cfg = Int4KvConfig(
        nbits=int(mcfg.get("nbits", 4)),
        group_size=int(mcfg.get("group_size", 32)),
    )
    max_new = int(args.max_new_tokens or cfg["decode"]["max_new_tokens"])

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

    # Warmup on first prompt (not scored into means).
    if prompts and args.warmup > 0:
        text = _apply_chat(tok, prompts[0].messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
            model.device
        )
        _timed_decode_fullkv(model, tok, ids, max_new_tokens=min(8, max_new))

    rows_out: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    t_wall0 = time.time()
    for prompt, ex in zip(prompts, examples, strict=True):
        text = _apply_chat(tok, prompt.messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        budget = int(cfg["vllm"]["max_model_len"]) - max_new - 8
        if ids.numel() > budget:
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])
        ids = ids.to(model.device)
        roles = assign_token_roles(tok, prompt.messages, chat_kwargs=chat_kwargs)

        full = _timed_decode_fullkv(model, tok, ids, max_new_tokens=max_new)
        full["example_id"] = prompt.id
        full["prompt_tokens"] = int(ids.numel())
        full["score"] = float(score_example(ex, full["text"]))
        rows_out.append(full)

        for policy in ("uniform", "structure"):
            fi = _timed_decode_fi(
                model,
                tok,
                ids,
                roles=roles,
                plan_cfg=plan_cfg,
                policy=policy,
                int4_cfg=int4_cfg,
                max_new_tokens=max_new,
            )
            fi["example_id"] = prompt.id
            fi["prompt_tokens"] = int(ids.numel())
            fi["score"] = float(score_example(ex, fi["text"]))
            rows_out.append(fi)
        print(
            f"[d4] {prompt.id} full_ttft={full['ttft_ms']:.1f}ms "
            f"full_tpot={full['tpot_ms']}",
            flush=True,
        )

    peak = int(torch.cuda.max_memory_allocated())
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows_out:
        by_arm.setdefault(r["arm"], []).append(r)

    def _mean(xs: list[float | None]) -> float | None:
        vals = [float(x) for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    summary_arms = {}
    for arm, rs in by_arm.items():
        summary_arms[arm] = {
            "n": len(rs),
            "ttft_ms_mean": _mean([r.get("ttft_ms") for r in rs]),
            "tpot_ms_mean": _mean([r.get("tpot_ms") for r in rs]),
            "tokens_per_s_mean": _mean([r.get("tokens_per_s") for r in rs]),
            "score_mean": _mean([r.get("score") for r in rs]),
            "int4_tokens_mean": _mean([r.get("int4_tokens") for r in rs]),
        }

    full_ttft = summary_arms.get("fullkv_sdpa", {}).get("ttft_ms_mean")
    struct_ttft = summary_arms.get("mixed_structure_fi_shim", {}).get("ttft_ms_mean")
    # Pass gate: ran cleanly + FI arms did not materialize + scores present.
    no_mat = all(not r.get("used_materialize_hf_past") for r in rows_out)
    decision = "D4_MICRO_PASS" if no_mat and summary_arms else "D4_MICRO_FAIL"
    result = {
        "job": "d4_latency",
        "tag": args.out_tag,
        "decision": decision,
        "pass": decision.endswith("PASS"),
        "n_examples": len(prompts),
        "max_new_tokens": max_new,
        "arms": summary_arms,
        "ttft_speedup_structure_vs_full": (
            (full_ttft / struct_ttft) if full_ttft and struct_ttft else None
        ),
        "cuda_peak_bytes": peak,
        "device": torch.cuda.get_device_name(0),
        "seconds": round(time.time() - t_wall0, 3),
        "rows": rows_out,
        "note": "Microbench on HF+FI-shim path; vLLM FP8 arm deferred to D4b.",
    }
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "pass": result["pass"],
                "arms": summary_arms,
                "ttft_speedup_structure_vs_full": result["ttft_speedup_structure_vs_full"],
                "cuda_peak_bytes": peak,
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
