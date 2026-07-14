"""HF generation with prompt-level sink+recent DropKeep (RoPE-safe)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from prioritykv.baselines.drop_keep import DropKeepConfig, apply_drop_keep_ids
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
        f"[dropkeep] mode=prompt_sink_recent sink={cfg.sink_tokens} "
        f"recent={cfg.recent_tokens} keep_tokens={cfg.keep_tokens} n={len(prompts)}",
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
            # Prefer keeping head (schemas) + tail (FINAL) already under max len.
            head = ids[: budget // 4]
            tail = ids[-(budget - int(head.numel())) :]
            ids = torch.cat([head, tail])

        ids_c, meta_keep = apply_drop_keep_ids(ids, cfg)
        inputs = {
            "input_ids": ids_c.unsqueeze(0).to(model.device),
            "attention_mask": torch.ones(1, ids_c.numel(), dtype=torch.long, device=model.device),
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
            "mode": "prompt_sink_recent",
            **meta_keep,
            "preview": new_text[:120],
            "seconds": time.time() - t0,
        }
        out.append((new_text, new_ids, meta))
        if tqdm is None:
            print(
                f"[dropkeep] {i + 1}/{len(prompts)} kept={meta['kept_tokens']}/"
                f"{meta['prompt_tokens']} {meta['seconds']:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                kept=f"{meta['kept_tokens']}/{meta['prompt_tokens']}",
                dropped=str(meta["dropped"]),
            )

    del model
    torch.cuda.empty_cache()
    print(f"[dropkeep] finished {len(out)}", flush=True)
    return out
