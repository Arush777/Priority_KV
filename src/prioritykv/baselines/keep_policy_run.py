"""Matched-budget structured vs uniform vs random keep (H200 quality job)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from prioritybench.pins import qwen3_chat_template_kwargs
from prioritykv.baselines.keep_policy import (
    KeepPolicyConfig,
    apply_keep_indices,
    assign_token_roles,
    select_keep_indices,
)
from prioritykv.fullkv_compare import PromptRow, _apply_chat


def run_transformers_keep_policy(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    policy: str,
    keep_cfg: KeepPolicyConfig,
    revision: str | None = None,
    max_model_len: int = 32768,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """policy ∈ {uniform, structure, structure_risk, fixed_hot, random, keep_all}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prioritykv.linear_risk import LinearRiskConfig, load_linear_risk_config

    risk_cfg: LinearRiskConfig | None = None
    if policy == "structure_risk":
        if keep_cfg.risk_fit_path:
            risk_cfg = load_linear_risk_config(keep_cfg.risk_fit_path)
        else:
            risk_cfg = LinearRiskConfig()
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
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **load_kwargs
        )
    model.eval()
    chat_kwargs = dict(qwen3_chat_template_kwargs())
    print(
        f"[keep] policy={policy} gran={keep_cfg.granularity} "
        f"keep_frac={keep_cfg.keep_frac} page_tokens={keep_cfg.page_tokens} "
        f"sink={keep_cfg.sink_tokens} force_recent={keep_cfg.force_recent} n={len(prompts)}",
        flush=True,
    )

    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore
    iterator = tqdm(prompts, desc=f"Keep[{policy}]", unit="ex") if tqdm else prompts

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

        if policy == "keep_all":
            kept_ids = ids
            meta_k = {
                "kept_tokens": n,
                "prompt_tokens": n,
                "keep_frac_realized": 1.0,
                "granularity": keep_cfg.granularity,
            }
        else:
            roles = None
            if policy in ("structure", "structure_risk"):
                roles = assign_token_roles(tok, row.messages, chat_kwargs=chat_kwargs)
                if len(roles) != n:
                    raise RuntimeError(
                        f"{policy} role/token mismatch: roles={len(roles)} n={n} "
                        f"id={row.id} (prompt exceeded max_model_len trim path)"
                    )
            cfg_i = keep_cfg
            if policy == "random":
                cfg_i = KeepPolicyConfig(
                    **{**keep_cfg.__dict__, "seed": keep_cfg.seed + i}
                )
            idx = select_keep_indices(
                n, cfg_i, policy=policy, roles=roles, risk_cfg=risk_cfg
            )
            kept_ids = apply_keep_indices(ids, idx)
            meta_k = {
                "kept_tokens": int(kept_ids.numel()),
                "prompt_tokens": n,
                "keep_frac_realized": float(kept_ids.numel()) / max(n, 1),
                "approx_compression_x": n / max(int(kept_ids.numel()), 1),
                "granularity": keep_cfg.granularity,
                "page_tokens": keep_cfg.page_tokens,
            }

        inputs = {
            "input_ids": kept_ids.unsqueeze(0).to(model.device),
            "attention_mask": torch.ones(
                1, kept_ids.numel(), dtype=torch.long, device=model.device
            ),
        }
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_ids = gen[0, inputs["input_ids"].shape[-1] :].tolist()
        new_text = tok.decode(new_ids, skip_special_tokens=True)
        meta = {
            "mode": f"keep_{policy}",
            "policy": policy,
            **meta_k,
            "preview": new_text[:120],
            "seconds": time.time() - t0,
        }
        out.append((new_text, new_ids, meta))
        if tqdm is None:
            print(
                f"[keep/{policy}] {i + 1}/{len(prompts)} "
                f"kept={meta_k['kept_tokens']}/{n} {meta['seconds']:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(kept=f"{meta_k['kept_tokens']}/{n}")

    del model
    torch.cuda.empty_cache()
    print(f"[keep/{policy}] finished {len(out)}", flush=True)
    return out
