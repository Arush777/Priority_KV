"""Uniform INT4 generation path for PriorityBench quality pilots (Q2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from prioritykv.fullkv_compare import PromptRow, _apply_chat
from prioritykv.int4_kv import Int4KvConfig, fake_quant_roundtrip, make_quantized_cache


def _to_legacy(past) -> tuple:
    if past is None:
        return ()
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return tuple(past)


def _from_legacy(legacy: tuple):
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


def _fake_quant_legacy(legacy: tuple, cfg: Int4KvConfig) -> tuple:
    import torch

    out = []
    for k, v in legacy:
        k_np = fake_quant_roundtrip(k.detach().float().cpu().numpy(), cfg)
        v_np = fake_quant_roundtrip(v.detach().float().cpu().numpy(), cfg)
        out.append(
            (
                torch.from_numpy(k_np).to(device=k.device, dtype=k.dtype),
                torch.from_numpy(v_np).to(device=v.device, dtype=v.dtype),
            )
        )
    return tuple(out)


def run_transformers_int4(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    revision: str | None = None,
    max_model_len: int = 32768,
    cfg: Optional[Int4KvConfig] = None,
    prefer_quanto: bool = True,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """Greedy decode with uniform INT4 KV.

    Prefer Transformers QuantizedCache (quanto). If unavailable, fall back to
    post-prefill fake-quant of the prompt KV (decode tokens stay BF16) — a
    lower-bound stress signal for agent reliability under INT4 storage.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or Int4KvConfig()
    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    out: list[tuple[str, list[int], dict[str, Any]]] = []

    for row in prompts:
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
        meta: dict[str, Any] = {"mode": "unknown", "prompt_tokens": int(inputs["input_ids"].shape[-1])}

        with torch.no_grad():
            done = False
            if prefer_quanto and cfg.backend != "fake":
                cache = make_quantized_cache(max_cache_len=max_model_len, cfg=cfg)
                if cache is not None:
                    try:
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            past_key_values=cache,
                        )
                        new_ids = gen[0, inputs["input_ids"].shape[-1] :].tolist()
                        new_text = tok.decode(new_ids, skip_special_tokens=True)
                        meta["mode"] = "quanto_quantized_cache"
                        out.append((new_text, new_ids, meta))
                        done = True
                    except Exception as exc:  # noqa: BLE001
                        meta["quanto_error"] = str(exc)[:240]

            if not done:
                meta["mode"] = "fake_groupwise_prefill"
                pre = model(**inputs, use_cache=True, return_dict=True)
                legacy = _fake_quant_legacy(_to_legacy(pre.past_key_values), cfg)
                past = _from_legacy(legacy)
                next_id = int(torch.argmax(pre.logits[:, -1, :], dim=-1).item())
                gen_ids = [next_id]
                attn = inputs["attention_mask"]
                cur = torch.tensor([[next_id]], device=model.device)
                for _ in range(max_new_tokens - 1):
                    attn = torch.cat(
                        [
                            attn,
                            torch.ones((1, 1), device=model.device, dtype=attn.dtype),
                        ],
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
                out.append((new_text, gen_ids, meta))

    del model
    torch.cuda.empty_cache()
    return out
