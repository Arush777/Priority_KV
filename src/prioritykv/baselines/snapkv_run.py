"""HF generation with kvpress SnapKVPress (matched-byte eviction pilot)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from prioritykv.baselines.snapkv import SnapKVConfig, make_press
from prioritykv.fullkv_compare import PromptRow, _apply_chat


def run_transformers_snapkv(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    revision: str | None = None,
    max_model_len: int = 32768,
    cfg: Optional[SnapKVConfig] = None,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """Generate with SnapKVPress hooks during prefill. Raises if press missing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or SnapKVConfig()
    press = make_press(cfg)
    if press is None:
        raise RuntimeError(
            "SnapKVPress unavailable — run `uv sync --extra kvpress` or "
            "LOCK_Q_DROPKEEP via scripts/run_snapkv_attempt.py"
        )

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
        f"[snapkv] compression_ratio={cfg.compression_ratio} "
        f"window={cfg.window_size} kernel={cfg.kernel_size} n={len(prompts)}",
        flush=True,
    )
    out: list[tuple[str, list[int], dict[str, Any]]] = []
    for i, pr in enumerate(prompts):
        text = _apply_chat(tok, pr.messages)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        n = int(input_ids.shape[-1])
        if n > max_model_len - max_new_tokens:
            input_ids = input_ids[:, -(max_model_len - max_new_tokens) :]
            n = int(input_ids.shape[-1])
        with torch.inference_mode():
            with press(model):
                gen = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        new_ids = gen[0, n:].tolist()
        txt = tok.decode(new_ids, skip_special_tokens=True)
        keep_frac = max(0.0, 1.0 - float(cfg.compression_ratio))
        meta = {
            "mode": "snapkv_press",
            "prompt_tokens": n,
            "compression_ratio": float(cfg.compression_ratio),
            "keep_frac_target": keep_frac,
            "window_size": int(cfg.window_size),
            "kernel_size": int(cfg.kernel_size),
        }
        out.append((txt, new_ids, meta))
        if (i + 1) % 1 == 0:
            print(f"[snapkv] {i + 1}/{len(prompts)} prompt_tokens={n}", flush=True)
    print(f"[snapkv] finished {len(out)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out
