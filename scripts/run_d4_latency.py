#!/usr/bin/env python3
"""D4 microbench M1: honest TTFT/TPOT with FI warmup + phase timing.

Fable 2026-07-18: prior TTFT mixed setup cost (lazy cold dequant / first FI call)
into the first decode step. This harness:

* times prefill / pack / cold_scratch / first_decode separately
* e2e_ttft_ms = prefill + pack + cold_scratch + first_decode
* decode_ttft_ms = first decode only (after eager_prepare_decode)
* one untimed FI+FullKV warmup before measured examples
* gate: mean decode_ttft(structure) <= 3× mean decode_ttft(fullkv)

H200: jobs/pending. Prints out=… for worker push.
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

from prioritybench.scoring import score_example
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


def _median(xs: list[float | None]) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _text_for_score(tok, token_ids: list[int], text: str) -> str:
    """Score the answer prefix only — fixed-length TPOT decode continues past EOS."""
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None and eos in token_ids:
        cut = token_ids.index(eos)
        return tok.decode(token_ids[:cut], skip_special_tokens=True)
    # Qwen3 sometimes emits thinking end markers when forced past stop.
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    return text


def _arm_summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
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


def _timed_decode_fullkv(model, tok, ids, *, max_new_tokens: int) -> dict[str, Any]:
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
            # Latency harness: do not stop on EOS — need fixed-length TPOT.

    prefill_ms = (t_pre1 - t_pre0) * 1000.0
    decode_ttft_ms = (t1 - t0) * 1000.0
    tpot_ms = (sum(step_ms) / len(step_ms)) if step_ms else None
    return {
        "arm": "fullkv_sdpa",
        "prefill_ms": prefill_ms,
        "pack_ms": 0.0,
        "cold_scratch_ms": 0.0,
        "decode_ttft_ms": decode_ttft_ms,
        "e2e_ttft_ms": prefill_ms + decode_ttft_ms,
        "ttft_ms": decode_ttft_ms,  # alias: decode-only, comparable across arms
        "tpot_ms": tpot_ms,
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
        _cuda_sync()
        t_pre1 = time.perf_counter()

        t_pack0 = time.perf_counter()
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
        t_pack1 = time.perf_counter()

        t_cold0 = time.perf_counter()
        eager_prepare_decode(state)
        _cuda_sync()
        t_cold1 = time.perf_counter()

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
                # Latency harness: do not stop on EOS — need fixed-length TPOT.
            used_mat = bool(ctx.used_materialize)
        state.assert_no_materialize_path(used_mat)

    prefill_ms = (t_pre1 - t_pre0) * 1000.0
    pack_ms = (t_pack1 - t_pack0) * 1000.0
    cold_ms = (t_cold1 - t_cold0) * 1000.0
    decode_ttft_ms = (t1 - t0) * 1000.0
    tpot_ms = (sum(step_ms) / len(step_ms)) if step_ms else None
    return {
        "arm": f"mixed_{policy}_fi_shim",
        "prefill_ms": prefill_ms,
        "pack_ms": pack_ms,
        "cold_scratch_ms": cold_ms,
        "decode_ttft_ms": decode_ttft_ms,
        "e2e_ttft_ms": prefill_ms + pack_ms + cold_ms + decode_ttft_ms,
        "ttft_ms": decode_ttft_ms,
        "tpot_ms": tpot_ms,
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
    ap.add_argument("--config", default=str(ROOT / "configs" / "d4_latency_micro.yaml"))
    ap.add_argument("--out-tag", default="m1_r1")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1, help="Timed repeats/example; median timings")
    ap.add_argument("--ttft-gate-mult", type=float, default=3.0)
    ap.add_argument("--m2-gate", action="store_true", help="Apply Fable M2 pack/cold/e2e gates")
    ap.add_argument("--m3-gate", action="store_true", help="Apply Fable M3 per-ctx gates")
    ap.add_argument("--pack-ms-max", type=float, default=200.0)
    ap.add_argument("--cold-ms-max", type=float, default=100.0)
    ap.add_argument("--e2e-gate-mult", type=float, default=1.15)
    ap.add_argument("--tpot-gate-mult", type=float, default=1.25)
    ap.add_argument("--pack-ms-max-16k", type=float, default=400.0)
    ap.add_argument("--cold-ms-max-16k", type=float, default=200.0)
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
    # Dual-GPU parent can restrict shard via env.
    only_ctx = os.environ.get("PRIORITYKV_ONLY_CONTEXT_LENGTH")
    if only_ctx:
        cfg = dict(cfg)
        sel = dict(cfg.get("selection") or {})
        sel["context_lengths"] = [int(only_ctx)]
        cfg["selection"] = sel
        print(f"[d4] ONLY_CONTEXT_LENGTH={only_ctx}", flush=True)
    bench = json.loads((ROOT / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    print(f"[d4] selected n={len(rows)} contexts={sorted({int(r['context_length']) for r in rows})}", flush=True)
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

    def _ids_for(prompt: PromptRow):
        text = _apply_chat(tok, prompt.messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        budget = int(cfg["vllm"]["max_model_len"]) - max_new - 8
        if ids.numel() > budget:
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])
        return ids.to(model.device)

    # Untimed warmup: FullKV + FI structure (JIT / cold scratch / kernels).
    if prompts and args.warmup > 0:
        print("[d4/m1] warmup FullKV + FI structure (untimed)…", flush=True)
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
        _timed_decode_fi(
            model,
            tok,
            w_ids,
            roles=w_roles,
            plan_cfg=plan_cfg,
            policy="uniform",
            int4_cfg=int4_cfg,
            max_new_tokens=min(8, max_new),
        )
        print("[d4/m1] warmup done", flush=True)

    rows_out: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    t_wall0 = time.time()
    repeats = max(1, int(args.repeats))
    timing_keys = (
        "prefill_ms",
        "pack_ms",
        "cold_scratch_ms",
        "decode_ttft_ms",
        "e2e_ttft_ms",
        "tpot_ms",
        "tokens_per_s",
        "ttft_ms",
    )

    def _run_arm(kind: str, *, policy: str | None = None) -> dict[str, Any]:
        reps: list[dict[str, Any]] = []
        for _ in range(repeats):
            if kind == "fullkv":
                reps.append(
                    _timed_decode_fullkv(model, tok, ids, max_new_tokens=max_new)
                )
            else:
                assert policy is not None
                reps.append(
                    _timed_decode_fi(
                        model,
                        tok,
                        ids,
                        roles=roles,
                        plan_cfg=plan_cfg,
                        policy=policy,
                        int4_cfg=int4_cfg,
                        max_new_tokens=max_new,
                    )
                )
        out = dict(reps[-1])
        for k in timing_keys:
            out[k] = _median([r.get(k) for r in reps])
        out["repeats"] = repeats
        out["example_id"] = prompt.id
        out["prompt_tokens"] = int(ids.numel())
        out["context_length"] = int(getattr(ex, "context_length", 0) or 0)
        if not out["context_length"]:
            # Fallback: parse from example_id …__c8000__…
            for part in str(prompt.id).split("__"):
                if part.startswith("c") and part[1:].isdigit():
                    out["context_length"] = int(part[1:])
                    break
        out["score"] = float(
            score_example(ex, _text_for_score(tok, list(out.get("token_ids") or []), out["text"]))
        )
        out["category"] = getattr(ex, "category", None)
        return out

    for prompt, ex in zip(prompts, examples, strict=True):
        ids = _ids_for(prompt)
        roles = assign_token_roles(tok, prompt.messages, chat_kwargs=chat_kwargs)

        full = _run_arm("fullkv")
        rows_out.append(full)
        for policy in ("uniform", "structure"):
            rows_out.append(_run_arm("fi", policy=policy))
        print(
            f"[d4] {prompt.id} ctx={full.get('context_length')} "
            f"full_e2e={full.get('e2e_ttft_ms'):.1f}ms repeats={repeats}",
            flush=True,
        )

    peak = int(torch.cuda.max_memory_allocated())
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows_out:
        by_arm.setdefault(r["arm"], []).append(r)

    summary_arms = {arm: _arm_summary(rs) for arm, rs in by_arm.items()}

    # Per-context summaries (M3).
    by_ctx: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows_out:
        ctx = str(int(r.get("context_length") or 0))
        by_ctx.setdefault(ctx, {}).setdefault(r["arm"], []).append(r)
    summary_by_ctx = {
        ctx: {arm: _arm_summary(rs) for arm, rs in arms.items()}
        for ctx, arms in by_ctx.items()
    }

    full_dec = summary_arms.get("fullkv_sdpa", {}).get("decode_ttft_ms_mean")
    struct = summary_arms.get("mixed_structure_fi_shim", {})
    struct_dec = struct.get("decode_ttft_ms_mean")
    full_e2e = summary_arms.get("fullkv_sdpa", {}).get("e2e_ttft_ms_mean")
    struct_e2e = struct.get("e2e_ttft_ms_mean")
    struct_pack = struct.get("pack_ms_mean")
    struct_cold = struct.get("cold_scratch_ms_mean")
    struct_tpot = struct.get("tpot_ms_mean")
    full_tpot = summary_arms.get("fullkv_sdpa", {}).get("tpot_ms_mean")
    no_mat = all(not r.get("used_materialize_hf_past") for r in rows_out)
    ttft_ratio = (struct_dec / full_dec) if full_dec and struct_dec else None
    e2e_ratio = (struct_e2e / full_e2e) if full_e2e and struct_e2e else None
    tpot_ratio = (struct_tpot / full_tpot) if full_tpot and struct_tpot else None
    ttft_gate = (
        ttft_ratio is not None and ttft_ratio <= float(args.ttft_gate_mult)
    )
    m2_pack_ok = struct_pack is not None and struct_pack <= float(args.pack_ms_max)
    m2_cold_ok = struct_cold is not None and struct_cold <= float(args.cold_ms_max)
    m2_e2e_ok = e2e_ratio is not None and e2e_ratio <= float(args.e2e_gate_mult)

    m3_ctx_gates: dict[str, Any] = {}
    m3_all_ok = True
    if args.m3_gate:
        for ctx, arms in summary_by_ctx.items():
            L = int(ctx)
            pack_lim = float(args.pack_ms_max if L <= 8000 else args.pack_ms_max_16k)
            cold_lim = float(args.cold_ms_max if L <= 8000 else args.cold_ms_max_16k)
            full_c = arms.get("fullkv_sdpa", {})
            st_c = arms.get("mixed_structure_fi_shim", {})
            dec_r = (
                (st_c.get("decode_ttft_ms_mean") / full_c.get("decode_ttft_ms_mean"))
                if full_c.get("decode_ttft_ms_mean") and st_c.get("decode_ttft_ms_mean")
                else None
            )
            e2e_r = (
                (st_c.get("e2e_ttft_ms_mean") / full_c.get("e2e_ttft_ms_mean"))
                if full_c.get("e2e_ttft_ms_mean") and st_c.get("e2e_ttft_ms_mean")
                else None
            )
            tpot_r = (
                (st_c.get("tpot_ms_mean") / full_c.get("tpot_ms_mean"))
                if full_c.get("tpot_ms_mean") and st_c.get("tpot_ms_mean")
                else None
            )
            pack_ok = st_c.get("pack_ms_mean") is not None and st_c["pack_ms_mean"] <= pack_lim
            cold_ok = (
                st_c.get("cold_scratch_ms_mean") is not None
                and st_c["cold_scratch_ms_mean"] <= cold_lim
            )
            dec_ok = dec_r is not None and dec_r <= float(args.ttft_gate_mult)
            e2e_ok = e2e_r is not None and e2e_r <= float(args.e2e_gate_mult)
            tpot_ok = tpot_r is not None and tpot_r <= float(args.tpot_gate_mult)
            st_score = float(st_c.get("score_mean") or 0.0)
            full_score = float(full_c.get("score_mean") or 0.0)
            # Relative to FullKV — 16k bench artifacts fail all arms equally.
            score_ok = st_score >= full_score - 0.01
            ctx_ok = bool(
                pack_ok and cold_ok and dec_ok and e2e_ok and tpot_ok and score_ok
            )
            m3_all_ok = m3_all_ok and ctx_ok
            m3_ctx_gates[ctx] = {
                "pack_lim": pack_lim,
                "cold_lim": cold_lim,
                "pack_ok": pack_ok,
                "cold_ok": cold_ok,
                "decode_ttft_ratio": dec_r,
                "e2e_ratio": e2e_r,
                "tpot_ratio": tpot_r,
                "decode_ok": dec_ok,
                "e2e_ok": e2e_ok,
                "tpot_ok": tpot_ok,
                "score_ok": score_ok,
                "pass": ctx_ok,
                "structure": {
                    "pack_ms": st_c.get("pack_ms_mean"),
                    "cold_ms": st_c.get("cold_scratch_ms_mean"),
                    "e2e_ms": st_c.get("e2e_ttft_ms_mean"),
                    "tpot_ms": st_c.get("tpot_ms_mean"),
                    "score": st_c.get("score_mean"),
                },
            }

    if args.m3_gate:
        if no_mat and m3_all_ok:
            decision = "D4_M3_PASS"
        elif no_mat:
            decision = "D4_M3_GATE_FAIL"
        else:
            decision = "D4_M3_FAIL"
        pass_ok = decision == "D4_M3_PASS"
        job_name = "d4_latency_m3"
    elif args.m2_gate:
        if no_mat and ttft_gate and m2_pack_ok and m2_cold_ok and m2_e2e_ok:
            decision = "D4_M2_PASS"
        elif no_mat and ttft_gate:
            decision = "D4_M2_E2E_GATE_FAIL"
        elif no_mat:
            decision = "D4_M2_TTFT_GATE_FAIL"
        else:
            decision = "D4_M2_FAIL"
        pass_ok = decision == "D4_M2_PASS"
        job_name = "d4_latency_m2"
    else:
        if no_mat and ttft_gate:
            decision = "D4_M1_PASS"
        elif no_mat:
            decision = "D4_M1_TTFT_GATE_FAIL"
        else:
            decision = "D4_M1_FAIL"
        pass_ok = decision == "D4_M1_PASS"
        job_name = "d4_latency_m1"

    result = {
        "job": job_name,
        "tag": args.out_tag,
        "decision": decision,
        "pass": pass_ok,
        "ttft_gate_mult": args.ttft_gate_mult,
        "m2_gate": bool(args.m2_gate),
        "m3_gate": bool(args.m3_gate),
        "decode_ttft_ratio_structure_vs_full": ttft_ratio,
        "e2e_ttft_ratio_structure_vs_full": e2e_ratio,
        "tpot_ratio_structure_vs_full": tpot_ratio,
        "m2": {
            "pack_ms_max": args.pack_ms_max,
            "cold_ms_max": args.cold_ms_max,
            "e2e_gate_mult": args.e2e_gate_mult,
            "pack_ok": m2_pack_ok,
            "cold_ok": m2_cold_ok,
            "e2e_ok": m2_e2e_ok,
            "structure_pack_ms": struct_pack,
            "structure_cold_ms": struct_cold,
        },
        "m3": {"ctx_gates": m3_ctx_gates, "all_ok": m3_all_ok if args.m3_gate else None},
        "n_examples": len(prompts),
        "max_new_tokens": max_new,
        "warmup": args.warmup,
        "repeats": repeats,
        "arms": summary_arms,
        "by_context": summary_by_ctx,
        "cuda_peak_bytes": peak,
        "device": torch.cuda.get_device_name(0),
        "seconds": round(time.time() - t_wall0, 3),
        "rows": rows_out,
        "note": (
            "M3: median over --repeats; e2e=prefill+pack+cold+decode; "
            "batched GPU pack (M2b)."
        ),
    }
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "pass": result["pass"],
                "decode_ttft_ratio_structure_vs_full": ttft_ratio,
                "e2e_ttft_ratio_structure_vs_full": e2e_ratio,
                "tpot_ratio_structure_vs_full": tpot_ratio,
                "m3": result["m3"],
                "by_context": {
                    ctx: {
                        arm: {
                            "e2e_ttft_ms_mean": v.get("e2e_ttft_ms_mean"),
                            "pack_ms_mean": v.get("pack_ms_mean"),
                            "cold_scratch_ms_mean": v.get("cold_scratch_ms_mean"),
                            "tpot_ms_mean": v.get("tpot_ms_mean"),
                            "score_mean": v.get("score_mean"),
                        }
                        for arm, v in arms.items()
                    }
                    for ctx, arms in summary_by_ctx.items()
                },
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
    print(f"out={out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
