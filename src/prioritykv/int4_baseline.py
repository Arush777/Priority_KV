"""Uniform INT4 generation path for PriorityBench quality pilots (Q2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import time

from prioritykv.fullkv_compare import PromptRow, _apply_chat
from prioritykv.int4_kv import Int4KvConfig, fake_quant_roundtrip, make_quantized_cache


def _fq_tensor(t, cfg: Int4KvConfig):
    import torch

    y = fake_quant_roundtrip(t.detach().float().cpu().numpy(), cfg)
    return torch.from_numpy(y).to(device=t.device, dtype=t.dtype)


def _fake_quant_past(past, cfg: Int4KvConfig):
    """INT4 round-trip K/V in whatever cache object HF returned."""
    import torch

    if past is None:
        return past

    # Newer transformers: Cache with .layers[*].keys / .values (or key_cache)
    layers = getattr(past, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr_k, attr_v in (
                ("keys", "values"),
                ("key_cache", "value_cache"),
                ("key", "value"),
            ):
                k = getattr(layer, attr_k, None)
                v = getattr(layer, attr_v, None)
                if k is None or v is None:
                    continue
                if torch.is_tensor(k) and torch.is_tensor(v):
                    setattr(layer, attr_k, _fq_tensor(k, cfg))
                    setattr(layer, attr_v, _fq_tensor(v, cfg))
                    break
                if isinstance(k, list) and isinstance(v, list):
                    for i in range(len(k)):
                        if k[i] is not None and torch.is_tensor(k[i]):
                            k[i] = _fq_tensor(k[i], cfg)
                        if v[i] is not None and torch.is_tensor(v[i]):
                            v[i] = _fq_tensor(v[i], cfg)
                    break
        return past

    # Older DynamicCache: parallel key_cache / value_cache lists
    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(len(kc)):
            if kc[i] is not None and torch.is_tensor(kc[i]):
                kc[i] = _fq_tensor(kc[i], cfg)
            if vc[i] is not None and torch.is_tensor(vc[i]):
                vc[i] = _fq_tensor(vc[i], cfg)
        return past

    # Legacy tuple/list of layer entries
    if isinstance(past, (tuple, list)):
        out = []
        for layer in past:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                if torch.is_tensor(k) and torch.is_tensor(v):
                    k2, v2 = _fq_tensor(k, cfg), _fq_tensor(v, cfg)
                    out.append((k2, v2) + tuple(layer[2:]))
                else:
                    out.append(layer)
            else:
                out.append(layer)
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(tuple(out))
        except Exception:
            return tuple(out)

    raise TypeError(f"unsupported past_key_values type: {type(past)}")


def run_transformers_int4(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    revision: str | None = None,
    max_model_len: int = 32768,
    cfg: Optional[Int4KvConfig] = None,
    prefer_quanto: bool = True,
    allow_fake_fallback: bool = True,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """Greedy decode with uniform INT4 KV.

    Prefer ``cache_implementation="quantized"`` (quanto). Fall back to
    post-prefill fake-quant of the prompt KV (decode tokens stay BF16)
    unless ``allow_fake_fallback=False`` (W3 assert mode — raise instead).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or Int4KvConfig()
    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    # Prefer dtype= over deprecated torch_dtype= when supported.
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
        f"[int4] model loaded; running {len(prompts)} prompts "
        f"(no vLLM bar here — per-example progress below)",
        flush=True,
    )

    out: list[tuple[str, list[int], dict[str, Any]]] = []
    cache_cfg = {
        "backend": "quanto",
        "nbits": int(cfg.nbits),
        "q_group_size": int(cfg.group_size),  # NOT group_size — HF QuantizedCache kw
    }

    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore

    iterator = (
        tqdm(prompts, desc="INT4 examples", unit="ex")
        if tqdm is not None
        else prompts
    )
    for i, row in enumerate(iterator):
        t_ex = time.time()
        if tqdm is None:
            print(f"[int4] example {i + 1}/{len(prompts)} id={row.id}", flush=True)
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
        meta: dict[str, Any] = {
            "mode": "unknown",
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        }

        with torch.no_grad():
            done = False
            if prefer_quanto and cfg.backend != "fake":
                # Path A: generate API with quantized cache impl (transformers ≥4.45)
                try:
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        cache_implementation="quantized",
                        cache_config=cache_cfg,
                    )
                    new_ids = gen[0, inputs["input_ids"].shape[-1] :].tolist()
                    new_text = tok.decode(new_ids, skip_special_tokens=True)
                    meta["mode"] = "hf_cache_implementation_quantized"
                    out.append((new_text, new_ids, meta))
                    done = True
                except Exception as exc:  # noqa: BLE001
                    meta["quanto_impl_error"] = str(exc)[:240]

                # Path B: explicit QuantizedCache object
                if not done:
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
                            meta["quanto_obj_error"] = str(exc)[:240]

            if not done:
                if not allow_fake_fallback:
                    errs = {
                        k: meta[k]
                        for k in ("quanto_impl_error", "quanto_obj_error")
                        if meta.get(k)
                    }
                    raise RuntimeError(
                        "INT4 quanto path failed and allow_fake_fallback=False "
                        f"(id={row.id} errors={errs})"
                    )
                meta["mode"] = "fake_groupwise_prefill"
                pre = model(**inputs, use_cache=True, return_dict=True)
                past = _fake_quant_past(pre.past_key_values, cfg)
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

        if tqdm is None:
            print(
                f"[int4] done {i + 1}/{len(prompts)} mode={meta['mode']} "
                f"tok={meta['prompt_tokens']} {time.time() - t_ex:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(mode=meta["mode"], tok=meta["prompt_tokens"])

    del model
    torch.cuda.empty_cache()
    print(f"[int4] finished {len(out)} examples", flush=True)
    return out
