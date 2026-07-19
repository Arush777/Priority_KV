"""HF generate under an arbitrary kvpress press (SnapKV / H2O / Pyramid / hybrid)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from prioritykv.fullkv_compare import PromptRow, _apply_chat


def _patch_kvpress_cache_position() -> None:
    """Inject missing ``cache_position`` (transformers×kvpress mismatch → KeyError)."""
    try:
        from kvpress.presses.base_press import BasePress  # type: ignore
    except Exception:
        return
    if getattr(BasePress, "_prioritykv_cache_position_patched", False):
        return

    import torch

    orig = BasePress.forward_hook

    def forward_hook(self, module, input, kwargs, output):  # noqa: A002
        kwargs = dict(kwargs)
        if kwargs.get("cache_position") is None:
            hs = kwargs.get("hidden_states")
            cache = kwargs.get("past_key_values")
            if hs is not None:
                q_len = int(hs.shape[1])
                device = hs.device
                seq = 0
                if cache is not None:
                    getter = getattr(cache, "get_seq_len", None)
                    if callable(getter):
                        try:
                            seq = int(getter())
                        except Exception:
                            seq = 0
                if seq > q_len:
                    # Decoding: fabricate absolute position so press skips compress.
                    kwargs["cache_position"] = torch.tensor([seq], device=device)
                else:
                    kwargs["cache_position"] = torch.arange(q_len, device=device)
        return orig(self, module, input, kwargs, output)

    BasePress.forward_hook = forward_hook  # type: ignore[method-assign]
    BasePress._prioritykv_cache_position_patched = True


def run_transformers_kvpress(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    press: Any,
    mode: str,
    revision: str | None = None,
    max_model_len: int = 32768,
    attn_implementation: str = "sdpa",
    per_prompt_press: Optional[Callable[[PromptRow, int, Any], Any]] = None,
    tokenizer_messages_hook: Optional[Callable[[PromptRow, Any, int], None]] = None,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """Generate with ``with press(model)``. ``per_prompt_press`` may swap press per example."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_kvpress_cache_position()

    if press is None and per_prompt_press is None:
        raise RuntimeError(f"{mode}: press is None (kvpress missing?)")

    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    load_kwargs: dict[str, Any] = {
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
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
        f"[kvpress/{mode}] attn={attn_implementation} n={len(prompts)}",
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

        use_press = press
        if per_prompt_press is not None:
            use_press = per_prompt_press(pr, n, tok)
        if use_press is None:
            raise RuntimeError(f"{mode}: per-prompt press returned None for {pr.id}")
        if tokenizer_messages_hook is not None:
            tokenizer_messages_hook(pr, tok, n)

        with torch.inference_mode():
            with use_press(model):
                gen = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        new_ids = gen[0, n:].tolist()
        txt = tok.decode(new_ids, skip_special_tokens=True)
        cr = float(getattr(use_press, "compression_ratio", -1.0))
        meta = {
            "mode": mode,
            "prompt_tokens": n,
            "compression_ratio": cr,
            "keep_frac_target": max(0.0, 1.0 - cr) if cr >= 0 else None,
            "attn_implementation": attn_implementation,
            "n_protected": len(getattr(use_press, "protected", ()) or ()),
        }
        out.append((txt, new_ids, meta))
        print(f"[kvpress/{mode}] {i + 1}/{len(prompts)} prompt_tokens={n}", flush=True)

    print(f"[kvpress/{mode}] finished {len(out)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out
