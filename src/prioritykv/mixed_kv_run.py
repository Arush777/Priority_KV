"""Real mixed-precision KV forward: per-position BF16/INT4 prompt cache (W6).

Unlike the keep experiments (which *drop* tokens and regenerate), this runs the
model on the full prompt, then quantizes only the INT4-planned prompt-KV
positions in-place before greedy decode. Structure positions stay BF16; low-risk
positions are round-tripped through INT4 (same groupwise error model as the green
uniform Q2 path). Decode tokens stay BF16.

This measures the *quality frontier* of role-aware mixed precision at a matched
byte budget (uniform vs structure). Wall-clock memory / latency (true packed
cache, FlashInfer) is a later stage; realized INT4 fraction is reported here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import prioritykv.cxx20_cuda_ext  # noqa: F401  — before any quanto/JIT touch
from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.fullkv_compare import PromptRow, _apply_chat
from prioritykv.int4_kv import Int4KvConfig, fake_quant_roundtrip
from prioritykv.linear_risk import LinearRiskConfig, load_linear_risk_config
from prioritykv.mixed_kv import MixedPlanConfig, plan_int4_mask
from prioritykv.page_roles import PageRole


def _fq_positions_tensor(t, idx, cfg: Int4KvConfig):
    """Round-trip KV at seq positions ``idx`` (axis=2) through INT4; rest untouched."""
    import torch

    if idx.numel() == 0 or not torch.is_tensor(t) or t.dim() != 4:
        return t
    sel = t.index_select(2, idx)  # (b, h, m, d)
    y = fake_quant_roundtrip(sel.detach().float().cpu().numpy(), cfg)
    y = torch.from_numpy(y).to(device=t.device, dtype=t.dtype)
    out = t.clone()
    out.index_copy_(2, idx, y)
    return out


def _fake_quant_positions(past, int4_mask: np.ndarray, cfg: Int4KvConfig):
    """Apply per-position INT4 round-trip to every layer's K/V prompt cache."""
    import torch

    if past is None:
        return past
    pos = np.nonzero(int4_mask)[0]
    if pos.size == 0:
        return past

    def _idx_for(t):
        s = t.shape[2]
        keep = pos[pos < s]
        return torch.as_tensor(keep, dtype=torch.long, device=t.device)

    layers = getattr(past, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr_k, attr_v in (("keys", "values"), ("key_cache", "value_cache"), ("key", "value")):
                k = getattr(layer, attr_k, None)
                v = getattr(layer, attr_v, None)
                if torch.is_tensor(k) and torch.is_tensor(v):
                    setattr(layer, attr_k, _fq_positions_tensor(k, _idx_for(k), cfg))
                    setattr(layer, attr_v, _fq_positions_tensor(v, _idx_for(v), cfg))
                    break
        return past

    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(len(kc)):
            if torch.is_tensor(kc[i]):
                kc[i] = _fq_positions_tensor(kc[i], _idx_for(kc[i]), cfg)
            if torch.is_tensor(vc[i]):
                vc[i] = _fq_positions_tensor(vc[i], _idx_for(vc[i]), cfg)
        return past

    raise TypeError(f"unsupported past_key_values type: {type(past)}")


def run_transformers_mixed_kv(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    policy: str,
    plan_cfg: MixedPlanConfig,
    int4_cfg: Optional[Int4KvConfig] = None,
    revision: str | None = None,
    max_model_len: int = 32768,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """policy ∈ {full, uniform, structure}. Greedy decode over a real mixed cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    int4_cfg = int4_cfg or Int4KvConfig()
    risk_cfg: LinearRiskConfig | None = None
    if policy == "structure":
        risk_cfg = (
            load_linear_risk_config(plan_cfg.risk_fit_path)
            if plan_cfg.risk_fit_path
            else LinearRiskConfig()
        )
    from prioritybench.pins import qwen3_chat_template_kwargs

    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    load_kwargs: dict[str, Any] = {
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    rev = revision if not Path(model_path).exists() else None
    if rev:
        load_kwargs["revision"] = rev
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **load_kwargs)
    model.eval()
    chat_kwargs = dict(qwen3_chat_template_kwargs())
    print(
        f"[mixed] policy={policy} int4_frac={plan_cfg.int4_frac} "
        f"sink={plan_cfg.sink_tokens} recent={plan_cfg.recent_window} n={len(prompts)}",
        flush=True,
    )

    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore
    iterator = tqdm(prompts, desc=f"Mixed[{policy}]", unit="ex") if tqdm else prompts

    out: list[tuple[str, list[int], dict[str, Any]]] = []
    for i, row in enumerate(iterator):
        t0 = time.time()
        text = _apply_chat(tok, row.messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        budget = max_model_len - max_new_tokens - 8
        if ids.numel() > budget:
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])
        n = int(ids.numel())
        inputs = {
            "input_ids": ids.unsqueeze(0).to(model.device),
            "attention_mask": torch.ones(1, n, dtype=torch.long, device=model.device),
        }

        int4_realized = 0
        with torch.no_grad():
            pre = model(**inputs, use_cache=True, return_dict=True)
            past = pre.past_key_values
            if policy != "full":
                roles = assign_token_roles(tok, row.messages, chat_kwargs=chat_kwargs)
                if len(roles) != n:
                    # Align to actual prompt length (no-trim case should match).
                    if len(roles) > n:
                        roles = roles[:n]
                    else:
                        roles = list(roles) + [PageRole.RECENT] * (n - len(roles))
                mask = plan_int4_mask(roles, plan_cfg, policy=policy, risk_cfg=risk_cfg)
                int4_realized = int(mask.sum())
                past = _fake_quant_positions(past, mask, int4_cfg)

            next_id = int(torch.argmax(pre.logits[:, -1, :], dim=-1).item())
            gen_ids = [next_id]
            attn = inputs["attention_mask"]
            cur = torch.tensor([[next_id]], device=model.device)
            for _ in range(max_new_tokens - 1):
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=model.device, dtype=attn.dtype)], dim=1
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
                gen_ids.append(nid)
                cur = torch.tensor([[nid]], device=model.device)
                if tok.eos_token_id is not None and nid == tok.eos_token_id:
                    break

        new_text = tok.decode(gen_ids, skip_special_tokens=True)
        meta = {
            "mode": f"mixed_{policy}",
            "policy": policy,
            "prompt_tokens": n,
            "int4_tokens": int4_realized,
            "int4_frac_realized": (int4_realized / n) if n else 0.0,
            "preview": new_text[:120],
            "seconds": time.time() - t0,
        }
        out.append((new_text, gen_ids, meta))
        if tqdm is None:
            print(
                f"[mixed/{policy}] {i + 1}/{len(prompts)} "
                f"int4={int4_realized}/{n} {meta['seconds']:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(int4=f"{int4_realized}/{n}")

    del model
    torch.cuda.empty_cache()
    print(f"[mixed/{policy}] finished {len(out)}", flush=True)
    return out
