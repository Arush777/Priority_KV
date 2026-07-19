"""Mixed-precision KV forward: packed BF16/INT4 prompt cache (W6→D3).

Unlike the keep experiments (which *drop* tokens and regenerate), this runs the
model on the prompt, packs INT4-planned positions into real ``PackedInt4Page``
storage, then materializes a dequantized HF cache for greedy decode.

``storage="packed"`` (default for ``degrade=int4``) is the systems path: demoted
positions leave BF16 tensors and live as INT4 codes + scales until materialize.
``storage="fake"`` keeps the legacy in-place round-trip for A/B parity.
``degrade="zero"`` zeroes demoted K/V in-place (planner/wiring stress).

``attn_backend="flashinfer"`` gates FlashInfer page-multicall + ``merge_state``
parity over the packed cache, then Stage-1b Qwen3 FI-shim decode (no
``materialize_hf_past``). ``attn_backend="sdpa"`` still materializes for SDPA.

Decode tokens stay BF16. Meta reports realized INT4 fraction + payload bytes.
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
from prioritykv.page_roles import PageRole, StorageDtype
from prioritykv.packed_mixed_cache import apply_packed_int4_to_hf_past


def _degrade_positions_tensor(t, idx, cfg: Int4KvConfig, *, degrade: str):
    """Degrade KV at seq positions ``idx`` (axis=2); rest untouched.

    ``degrade``:
      - ``int4``: groupwise fake-quant round-trip (legacy ``storage=fake``)
      - ``zero``: zero demoted K/V (INT0 stress — proves mask wiring / planner)
    """
    import torch

    if idx.numel() == 0 or not torch.is_tensor(t) or t.dim() != 4:
        return t
    out = t.clone()
    if degrade == "zero":
        out.index_fill_(2, idx, 0)
        return out
    if degrade != "int4":
        raise ValueError(f"unknown degrade mode {degrade}")
    sel = t.index_select(2, idx)  # (b, h, m, d)
    y = fake_quant_roundtrip(sel.detach().float().cpu().numpy(), cfg)
    y = torch.from_numpy(y).to(device=t.device, dtype=t.dtype)
    out.index_copy_(2, idx, y)
    return out


def _degrade_positions(
    past, int4_mask: np.ndarray, cfg: Int4KvConfig, *, degrade: str = "int4"
):
    """Apply per-position degrade to every layer's K/V prompt cache (in-place)."""
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
                    setattr(
                        layer,
                        attr_k,
                        _degrade_positions_tensor(k, _idx_for(k), cfg, degrade=degrade),
                    )
                    setattr(
                        layer,
                        attr_v,
                        _degrade_positions_tensor(v, _idx_for(v), cfg, degrade=degrade),
                    )
                    break
        return past

    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(len(kc)):
            if torch.is_tensor(kc[i]):
                kc[i] = _degrade_positions_tensor(
                    kc[i], _idx_for(kc[i]), cfg, degrade=degrade
                )
            if torch.is_tensor(vc[i]):
                vc[i] = _degrade_positions_tensor(
                    vc[i], _idx_for(vc[i]), cfg, degrade=degrade
                )
        return past

    raise TypeError(f"unsupported past_key_values type: {type(past)}")


def _resolve_storage(degrade: str, storage: str | None) -> str:
    """Default storage: packed for int4, none for zero/full."""
    if storage is not None:
        if storage not in ("packed", "fake"):
            raise ValueError(f"storage must be packed|fake, got {storage}")
        return storage
    if degrade == "int4":
        return "packed"
    return "fake"


def _resolve_attn_backend(attn_backend: str | None, storage_mode: str) -> str:
    """Default SDPA; flashinfer requires packed storage."""
    backend = (attn_backend or "sdpa").lower()
    if backend not in ("sdpa", "flashinfer"):
        raise ValueError(f"attn_backend must be sdpa|flashinfer, got {backend}")
    if backend == "flashinfer" and storage_mode != "packed":
        raise ValueError("attn_backend=flashinfer requires storage=packed")
    return backend


def run_transformers_mixed_kv(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    *,
    policy: str,
    plan_cfg: MixedPlanConfig,
    int4_cfg: Optional[Int4KvConfig] = None,
    degrade: str = "int4",
    storage: str | None = None,
    attn_backend: str | None = None,
    fi_parity_every: int = 1,
    fi_require_pass: bool = True,
    cold_attend: str = "full",
    cold_chunk_tokens: int = 1024,
    revision: str | None = None,
    max_model_len: int = 32768,
) -> list[tuple[str, list[int], dict[str, Any]]]:
    """policy ∈ {full, uniform, structure}. Greedy decode over a mixed cache.

    ``storage``:
      - ``packed`` (default when degrade=int4): true INT4 page payloads via
        ``PackedMixedCache``, then dequant materialize for SDPA decode.
      - ``fake``: legacy in-place groupwise round-trip (A/B parity).

    ``attn_backend``:
      - ``sdpa`` (default): Transformers SDPA on materialized KV.
      - ``flashinfer``: pack → FI parity gate → Qwen3 FI-shim greedy decode
        (no ``materialize_hf_past``).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if degrade not in ("int4", "zero"):
        raise ValueError(f"degrade must be int4|zero, got {degrade}")
    storage_mode = _resolve_storage(degrade, storage)
    if degrade == "zero" and storage_mode == "packed":
        raise ValueError("degrade=zero has no packed representation; use storage=fake")
    backend = _resolve_attn_backend(attn_backend, storage_mode)
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
        f"[mixed] policy={policy} degrade={degrade} storage={storage_mode} "
        f"attn={backend} int4_frac={plan_cfg.int4_frac} nbits={int4_cfg.nbits} "
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
        if n < 2:
            raise RuntimeError(f"prompt too short for split prefill: n={n} id={row.id}")
        ids = ids.to(model.device)
        # Correctness-critical split prefill:
        # 1) prefill all but the final prompt token,
        # 2) degrade/pack that cache,
        # 3) replay the final prompt token against the degraded cache.
        cache_n = n - 1
        prefill_inputs = {
            "input_ids": ids[:cache_n].unsqueeze(0),
            "attention_mask": torch.ones(
                1, cache_n, dtype=torch.long, device=model.device
            ),
        }

        int4_realized = 0
        packed_meta: dict[str, Any] = {}
        with torch.no_grad():
            pre = model(**prefill_inputs, use_cache=True, return_dict=True)
            past = pre.past_key_values
            if policy != "full":
                roles = assign_token_roles(tok, row.messages, chat_kwargs=chat_kwargs)
                if len(roles) != cache_n:
                    if len(roles) > cache_n:
                        roles = roles[:cache_n]
                    else:
                        roles = list(roles) + [PageRole.RECENT] * (
                            cache_n - len(roles)
                        )
                mask = plan_int4_mask(roles, plan_cfg, policy=policy, risk_cfg=risk_cfg)
                int4_realized = int(mask.sum())
                if storage_mode == "packed" and degrade == "int4":
                    if backend == "flashinfer":
                        # Stage-1b path: pack → FiMixedDecodeState → FI shim decode.
                        # Never call materialize_hf_past.
                        from prioritykv.flashinfer_multicall import (
                            verify_packed_cache_flashinfer,
                        )
                        from prioritykv.qwen3_fi_shim import (
                            FiSeqLenCache,
                            fi_shim_context,
                            pack_prefill_to_fi_state,
                        )

                        packed, fi_state = pack_prefill_to_fi_state(
                            past,
                            roles,
                            mask,
                            device=model.device,
                            dtype=torch.bfloat16,
                            int4_cfg=int4_cfg,
                            decode_tail_cap=max(max_new_tokens + 8, 64),
                            cold_attend=cold_attend,
                            cold_chunk_tokens=cold_chunk_tokens,
                        )
                        packed_meta = {
                            "storage": "packed",
                            "attn_backend": "flashinfer_fi_shim",
                            "cold_attend": cold_attend,
                            "cold_chunk_tokens": int(cold_chunk_tokens),
                            "n_pages": len(packed.page_manager.pages),
                            "payload_bytes": packed.payload_bytes(),
                            "realized_bytes": packed.realized_bytes(),
                            "fullkv_bf16_bytes": packed.fullkv_bf16_bytes(),
                            "compression_ratio": round(packed.compression_ratio(), 6),
                            "int4_tokens_pages": packed.dtype_token_counts()[
                                StorageDtype.INT4
                            ],
                            "used_materialize_hf_past": False,
                        }
                        if fi_parity_every > 0 and i % fi_parity_every == 0:
                            fi_res = verify_packed_cache_flashinfer(packed)
                            packed_meta["fi_parity"] = {
                                "decision": fi_res.get("decision"),
                                "pass": fi_res.get("pass"),
                                "layers": [
                                    {
                                        "layer": r.get("layer"),
                                        "decision": r.get("decision"),
                                        "err": r.get(
                                            "fi_multicall_vs_fi_dense_max_abs"
                                        ),
                                    }
                                    for r in fi_res.get("layers", [])
                                ],
                            }
                            print(
                                f"[mixed/{policy}] fi_parity={fi_res.get('decision')} "
                                f"pass={fi_res.get('pass')}",
                                flush=True,
                            )
                            if fi_require_pass and fi_res.get("pass") is False:
                                raise RuntimeError(
                                    f"FlashInfer packed parity failed: {fi_res}"
                                )

                        stub = FiSeqLenCache(fi_state)
                        attn = torch.ones(1, n, dtype=torch.long, device=model.device)
                        with fi_shim_context(fi_state) as ctx:
                            replay = model(
                                input_ids=ids[-1:].view(1, 1),
                                attention_mask=attn,
                                past_key_values=stub,
                                use_cache=True,
                                return_dict=True,
                            )
                            next_id = int(
                                torch.argmax(replay.logits[:, -1, :], dim=-1).item()
                            )
                            gen_ids = [next_id]
                            cur = torch.tensor([[next_id]], device=model.device)
                            for _ in range(max_new_tokens - 1):
                                attn = torch.cat(
                                    [
                                        attn,
                                        torch.ones(
                                            (1, 1), device=model.device, dtype=attn.dtype
                                        ),
                                    ],
                                    dim=1,
                                )
                                step = model(
                                    input_ids=cur,
                                    attention_mask=attn,
                                    past_key_values=stub,
                                    use_cache=True,
                                    return_dict=True,
                                )
                                nid = int(
                                    torch.argmax(step.logits[:, -1, :], dim=-1).item()
                                )
                                gen_ids.append(nid)
                                cur = torch.tensor([[nid]], device=model.device)
                                if (
                                    tok.eos_token_id is not None
                                    and nid == tok.eos_token_id
                                ):
                                    break
                            packed_meta["used_materialize_hf_past"] = bool(
                                ctx.used_materialize
                            )
                        fi_state.assert_no_materialize_path(
                            bool(packed_meta["used_materialize_hf_past"])
                        )
                        new_text = tok.decode(gen_ids, skip_special_tokens=True)
                        meta = {
                            "mode": f"mixed_{policy}",
                            "policy": policy,
                            "degrade": degrade,
                            "storage": packed_meta.get("storage", "packed"),
                            "attn_backend": "flashinfer_fi_shim",
                            "nbits": int(int4_cfg.nbits),
                            "prompt_tokens": n,
                            "cache_tokens_degraded": cache_n,
                            "int4_tokens": int4_realized,
                            "int4_frac_realized": (int4_realized / cache_n)
                            if cache_n
                            else 0.0,
                            "first_token_from_degraded_cache": True,
                            "preview": new_text[:120],
                            "seconds": time.time() - t0,
                            **{
                                k: v
                                for k, v in packed_meta.items()
                                if k not in ("storage", "attn_backend")
                            },
                        }
                        out.append((new_text, gen_ids, meta))
                        if tqdm is None:
                            print(
                                f"[mixed/{policy}] {i + 1}/{len(prompts)} "
                                f"int4={int4_realized}/{n} storage=packed "
                                f"attn=flashinfer_fi_shim {meta['seconds']:.1f}s",
                                flush=True,
                            )
                        elif hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(int4=f"{int4_realized}/{n}")
                        continue

                    past, packed = apply_packed_int4_to_hf_past(
                        past,
                        roles,
                        mask,
                        int4_cfg=int4_cfg,
                        device=model.device,
                        dtype=torch.bfloat16,
                    )
                    packed_meta = {
                        "storage": "packed",
                        "n_pages": len(packed.page_manager.pages),
                        "payload_bytes": packed.payload_bytes(),
                        "realized_bytes": packed.realized_bytes(),
                        "fullkv_bf16_bytes": packed.fullkv_bf16_bytes(),
                        "compression_ratio": round(packed.compression_ratio(), 6),
                        "int4_tokens_pages": packed.dtype_token_counts()[
                            StorageDtype.INT4
                        ],
                    }
                    if backend == "flashinfer" and (
                        fi_parity_every > 0 and i % fi_parity_every == 0
                    ):
                        from prioritykv.flashinfer_multicall import (
                            verify_packed_cache_flashinfer,
                        )

                        fi_res = verify_packed_cache_flashinfer(packed)
                        packed_meta["attn_backend"] = "flashinfer"
                        packed_meta["fi_parity"] = {
                            "decision": fi_res.get("decision"),
                            "pass": fi_res.get("pass"),
                            "layers": [
                                {
                                    "layer": r.get("layer"),
                                    "decision": r.get("decision"),
                                    "err": r.get("fi_multicall_vs_fi_dense_max_abs"),
                                }
                                for r in fi_res.get("layers", [])
                            ],
                        }
                        print(
                            f"[mixed/{policy}] fi_parity={fi_res.get('decision')} "
                            f"pass={fi_res.get('pass')}",
                            flush=True,
                        )
                        if fi_require_pass and fi_res.get("pass") is False:
                            raise RuntimeError(
                                f"FlashInfer packed parity failed: {fi_res}"
                            )
                else:
                    past = _degrade_positions(past, mask, int4_cfg, degrade=degrade)
                    packed_meta = {
                        "storage": storage_mode if degrade == "int4" else "inplace",
                        "attn_backend": "sdpa",
                    }
            else:
                packed_meta = {"storage": "none", "attn_backend": "sdpa"}

            # Recompute the first output-token logits from the degraded cache.
            attn = torch.ones(1, n, dtype=torch.long, device=model.device)
            replay = model(
                input_ids=ids[-1:].view(1, 1),
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = replay.past_key_values
            next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
            gen_ids = [next_id]
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
            "degrade": degrade if policy != "full" else "none",
            "storage": packed_meta.get("storage", "none"),
            "attn_backend": packed_meta.get("attn_backend", backend if policy != "full" else "sdpa"),
            "nbits": int(int4_cfg.nbits) if policy != "full" and degrade == "int4" else None,
            "prompt_tokens": n,
            "cache_tokens_degraded": cache_n,
            "int4_tokens": int4_realized,
            "int4_frac_realized": (int4_realized / cache_n) if cache_n else 0.0,
            "first_token_from_degraded_cache": True,
            "preview": new_text[:120],
            "seconds": time.time() - t0,
            **{
                k: v
                for k, v in packed_meta.items()
                if k not in ("storage", "attn_backend")
            },
        }
        out.append((new_text, gen_ids, meta))
        if tqdm is None:
            print(
                f"[mixed/{policy}] {i + 1}/{len(prompts)} "
                f"int4={int4_realized}/{n} storage={meta.get('storage')} "
                f"attn={meta.get('attn_backend')} {meta['seconds']:.1f}s",
                flush=True,
            )
        elif hasattr(iterator, "set_postfix"):
            iterator.set_postfix(int4=f"{int4_realized}/{n}")

    del model
    torch.cuda.empty_cache()
    print(f"[mixed/{policy}] finished {len(out)}", flush=True)
    return out
