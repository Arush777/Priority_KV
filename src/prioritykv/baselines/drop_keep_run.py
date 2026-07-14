"""HF generation with aggressive DropKeep eviction (must show PriorityBench damage)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from prioritykv.baselines.drop_keep import DropKeepConfig, drop_keep_past, realized_keep_frac
from prioritykv.fullkv_compare import PromptRow, _apply_chat


def run_transformers_dropkeep(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    revision: str | None = None,
    max_model_len: int = 32768,
    cfg: Optional[DropKeepConfig] = None,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or DropKeepConfig()
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
    print(
        f"[dropkeep] sink={cfg.sink_tokens} recent={cfg.recent_tokens} "
        f"keep_tokens={cfg.keep_tokens} n={len(prompts)}",
        flush=True,
    )

    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore
    iterator = tqdm(prompts, desc="DropKeep", unit="ex") if tqdm else prompts

    out: list[tuple[str, list[int], dict[str, Any]]] = []
    for i, row in enumerate(iterator):
        t0 = time.time()
        text = _apply_chat(tok, row.messages)
        raw = tok(text, return_tensors="pt")
        budget = max_model_len - max_new_tokens - 8
        ids = raw["input_ids"][0]
        if ids.numel() > budget:
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])
            inputs = {
                "input_ids": ids.unsqueeze(0),
                "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
            }
        else:
            inputs = {k: v for k, v in raw.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        keep_frac = realized_keep_frac(prompt_len, cfg)

        with torch.no_grad():
            pre = model(**inputs, use_cache=True, return_dict=True)
            past = drop_keep_past(pre.past_key_values, cfg)

            def _past_len(p) -> int:
                layers = getattr(p, "layers", None)
                if layers:
                    for layer in layers:
                        for attr in ("keys", "key_cache", "key"):
                            t = getattr(layer, attr, None)
                            if t is not None and hasattr(t, "shape") and t.ndim >= 3:
                                return int(t.shape[-2])
                kc = getattr(p, "key_cache", None)
                if isinstance(kc, list) and kc and kc[0] is not None:
                    return int(kc[0].shape[-2])
                if isinstance(p, (tuple, list)) and p:
                    return int(p[0][0].shape[-2])
                return min(prompt_len, cfg.sink_tokens + cfg.recent_tokens)

            kept = _past_len(past)
            attn = torch.ones(
                (1, kept), device=model.device, dtype=inputs["attention_mask"].dtype
            )
            next_id = int(torch.argmax(pre.logits[:, -1, :], dim=-1).item())
            gen_ids = [next_id]
            cur = torch.tensor([[next_id]], device=model.device)
            for _ in range(max_new_tokens - 1):
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=model.device, dtype=attn.dtype)],
                    dim=1,
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
            "mode": "drop_keep_sink_recent",
            "prompt_tokens": prompt_len,
            "kept_tokens": kept,
            "keep_frac": kept / max(prompt_len, 1),
            "approx_compression_x": prompt_len / max(kept, 1),
            "seconds": time.time() - t0,
        }
        out.append((new_text, gen_ids, meta))
        if tqdm is None:
            print(
                f"[dropkeep] {i + 1}/{len(prompts)} keep_frac={keep_frac:.3f} "
                f"{meta['seconds']:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(keep=f"{keep_frac:.3f}")

    del model
    torch.cuda.empty_cache()
    print(f"[dropkeep] finished {len(out)}", flush=True)
    return out
