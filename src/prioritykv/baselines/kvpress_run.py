"""HF generate under an arbitrary kvpress press (SnapKV / H2O / Pyramid / hybrid)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from prioritykv.fullkv_compare import PromptRow, _apply_chat


def _patch_kvpress_cache_position() -> None:
    """Inject missing ``cache_position`` and skip compress on decode steps.

    transformers×kvpress often omit cache_position; without a skip, SnapKV
    asserts ``q_len > window_size`` on the first decode token (q_len=1).
    """
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
        hs = kwargs.get("hidden_states")
        if hs is None:
            return orig(self, module, input, kwargs, output)
        q_len = int(hs.shape[1])
        # Decode / short chunk: do not compress (fixes SnapKV window assert).
        window = int(getattr(self, "window_size", 0) or 0)
        if q_len <= 1 or (window > 0 and q_len <= window):
            return output
        if kwargs.get("cache_position") is None:
            cache = kwargs.get("past_key_values")
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
                return output
            kwargs["cache_position"] = torch.arange(q_len, device=device)
        return orig(self, module, input, kwargs, output)

    BasePress.forward_hook = forward_hook  # type: ignore[method-assign]
    BasePress._prioritykv_cache_position_patched = True


def _hf_device_map(
    device_map: str,
    *,
    max_memory_gib: float | None,
) -> tuple[Any, dict[int, str] | None]:
    """Resolve HF ``device_map`` / ``max_memory``.

    For eager H2O on long contexts, pass ``device_map=auto`` with a *low*
    ``max_memory_gib`` so weights split across visible GPUs and leave HBM for
    attention maps (``CUDA_VISIBLE_DEVICES`` already isolates the pair).
    """
    import torch

    n = int(torch.cuda.device_count())
    if device_map in ("", "cuda:0", "single") or (device_map == "auto" and n <= 1):
        return "cuda:0", None
    if device_map in ("auto", "balanced", "multi"):
        # Cap weight placement so auto *must* shard (Qwen3-8B bf16 ≈ 16 GiB).
        per = float(max_memory_gib) if max_memory_gib is not None else 12.0
        return "auto", {i: f"{per:.0f}GiB" for i in range(n)}
    return device_map, None


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
    device_map: str = "cuda:0",
    max_memory_gib: float | None = None,
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
    dm, max_mem = _hf_device_map(device_map, max_memory_gib=max_memory_gib)
    load_kwargs: dict[str, Any] = {
        "device_map": dm,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    if max_mem is not None:
        load_kwargs["max_memory"] = max_mem
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
    try:
        in_device = model.get_input_embeddings().weight.device
    except Exception:  # noqa: BLE001
        in_device = torch.device("cuda:0")

    print(
        f"[kvpress/{mode}] attn={attn_implementation} device_map={dm} "
        f"max_memory={max_mem} n_gpu={torch.cuda.device_count()} n={len(prompts)}",
        flush=True,
    )
    out: list[tuple[str, list[int], dict[str, Any]]] = []
    for i, pr in enumerate(prompts):
        text = _apply_chat(tok, pr.messages)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(in_device)
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
            "device_map": str(dm),
            "n_protected": len(getattr(use_press, "protected", ()) or ()),
        }
        out.append((txt, new_ids, meta))
        print(f"[kvpress/{mode}] {i + 1}/{len(prompts)} prompt_tokens={n}", flush=True)
        # Eager H2O builds huge attn maps — free between examples.
        if attn_implementation == "eager":
            torch.cuda.empty_cache()

    print(f"[kvpress/{mode}] finished {len(out)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out
