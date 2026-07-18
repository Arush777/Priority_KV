"""Qwen3 attention shim for Stage-1b FI decode (council 2026-07-17).

Design locks (Fable GO):
* Monkeypatch ``Qwen3Attention.forward`` on all layers — not an HF Cache subclass.
* Split-prefill: HF ``prefill(n-1)`` → pack → ``FiMixedDecodeState`` → FI replay
  of the last prompt token, then greedy decode.
* Per step: project q/k/v (q_norm/k_norm + RoPE) → ``append_decode_kv`` →
  FI attend ≤2 chunks → ``commit_decode_step`` after all layers.
* Refuse silent ``materialize_hf_past`` while the shim context is active.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from prioritykv.fi_mixed_decode import (
    FiMixedDecodeState,
    append_decode_kv,
    attend_layer_flashinfer,
    build_from_packed_cache,
    commit_decode_step,
)
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig, plan_int4_mask
from prioritykv.packed_mixed_cache import build_from_hf_prefill_batched
from prioritykv.page_roles import PageRole


@dataclass
class _ShimCtx:
    state: FiMixedDecodeState
    layers_seen: set[int] = field(default_factory=set)
    n_layers: int = 0
    used_materialize: bool = False


_ACTIVE: ContextVar[Optional[_ShimCtx]] = ContextVar("prioritykv_fi_shim", default=None)


class FiSeqLenCache:
    """Duck-typed past stub so HF computes RoPE positions from committed length.

    ``update`` must never run — the FI shim owns KV storage.
    Implements the Cache methods ``create_causal_mask`` calls
    (``get_seq_length``, ``get_mask_sizes``).
    """

    is_compileable = False

    def __init__(self, state: FiMixedDecodeState):
        self.state = state

    def get_seq_length(self, layer_idx: int = 0) -> int:  # noqa: ARG002
        return int(self.state.total_kv_len)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:  # noqa: ARG002
        """Match DynamicCache: kv_length = seen + query, offset 0."""
        return int(self.get_seq_length() + query_length), 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> int:
        return -1

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:  # noqa: ARG002
        return self.get_seq_length(layer_idx)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "FiSeqLenCache.update forbidden — Stage-1b FI shim owns decode KV"
        )

    def __len__(self) -> int:
        return int(self.state.num_layers)


def _fi_attention_forward(
    self: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    attention_mask: Any = None,  # noqa: ARG001
    past_key_values: Any = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> tuple[Any, None]:
    """Drop-in Qwen3Attention.forward that attends via FiMixedDecodeState."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    ctx = _ACTIVE.get()
    if ctx is None:
        raise RuntimeError("FI shim forward called without active context")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    tq = int(input_shape[1]) if len(input_shape) > 1 else int(hidden_states.shape[1])
    if tq != 1:
        raise RuntimeError(
            f"Stage-1b FI shim supports decode tq=1 only, got tq={tq}"
        )

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # HF layout after transpose: (batch, heads, tq, dim) → squeeze batch.
    if query_states.shape[0] != 1:
        raise RuntimeError("FI shim expects batch=1")
    q = query_states[0].transpose(0, 1).contiguous()  # (tq, qo_heads, dim)
    k = key_states[0]  # (kv_heads, tq, dim)
    v = value_states[0]

    layer_idx = int(self.layer_idx)
    append_decode_kv(ctx.state, layer_idx, k, v)
    ctx.layers_seen.add(layer_idx)

    attn = attend_layer_flashinfer(q, ctx.state, layer_idx, causal=False)
    # attn: (tq, qo_heads, dim) → (batch, tq, hidden)
    attn_output = attn.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if len(ctx.layers_seen) >= ctx.n_layers:
        commit_decode_step(ctx.state)
        ctx.layers_seen.clear()

    return attn_output, None


@contextlib.contextmanager
def fi_shim_context(state: FiMixedDecodeState) -> Iterator[_ShimCtx]:
    """Patch Qwen3Attention.forward + tripwire materialize_hf_past."""
    import prioritykv.packed_mixed_cache as pmc
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    ctx = _ShimCtx(state=state, n_layers=int(state.num_layers))
    token = _ACTIVE.set(ctx)
    orig_forward = Qwen3Attention.forward
    orig_materialize = pmc.materialize_hf_past

    def _materialize_boom(*_a: Any, **_k: Any) -> Any:
        ctx.used_materialize = True
        state.assert_no_materialize_path(True)
        raise RuntimeError("unreachable")  # pragma: no cover

    Qwen3Attention.forward = _fi_attention_forward  # type: ignore[method-assign]
    pmc.materialize_hf_past = _materialize_boom  # type: ignore[assignment]
    try:
        yield ctx
    finally:
        Qwen3Attention.forward = orig_forward  # type: ignore[method-assign]
        pmc.materialize_hf_past = orig_materialize  # type: ignore[assignment]
        _ACTIVE.reset(token)


def pack_prefill_to_fi_state(
    past: Any,
    roles: Sequence[PageRole],
    int4_mask: Any,
    *,
    device: Any,
    dtype: Any = None,
    int4_cfg: Optional[Int4KvConfig] = None,
    decode_tail_cap: int = 256,
) -> tuple[Any, FiMixedDecodeState]:
    """Pack HF prefill past → PackedMixedCache → FiMixedDecodeState (no materialize).

    M2b: batched gather+quantize (≤1 BF16 page + ≤1 INT4 page per layer).
    ``roles`` retained for API compat; mask drives packing.
    """
    import torch

    dtype = dtype or torch.bfloat16
    cfg = int4_cfg or Int4KvConfig()
    _ = roles  # API compat / callers still pass roles for planning upstream
    packed = build_from_hf_prefill_batched(past, int4_mask, int4_cfg=cfg)
    state = build_from_packed_cache(
        packed, device=device, dtype=dtype, decode_tail_cap=decode_tail_cap
    )
    state.assert_no_materialize_path(False)
    return packed, state


def greedy_fi_decode(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    *,
    roles: Sequence[PageRole],
    plan_cfg: MixedPlanConfig,
    policy: str = "structure",
    max_new_tokens: int = 8,
    int4_cfg: Optional[Int4KvConfig] = None,
    decode_tail_cap: int = 256,
) -> dict[str, Any]:
    """Split-prefill + FI-shim greedy decode. Returns tokens + meta."""
    import torch

    if input_ids.dim() == 1:
        ids = input_ids
    else:
        ids = input_ids[0]
    n = int(ids.numel())
    if n < 2:
        raise RuntimeError(f"prompt too short for split prefill: n={n}")
    device = model.device
    ids = ids.to(device)
    cache_n = n - 1
    int4_cfg = int4_cfg or Int4KvConfig()

    with torch.no_grad():
        pre = model(
            input_ids=ids[:cache_n].unsqueeze(0),
            attention_mask=torch.ones(1, cache_n, dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        past = pre.past_key_values
        role_list = list(roles)
        if len(role_list) != cache_n:
            if len(role_list) > cache_n:
                role_list = role_list[:cache_n]
            else:
                role_list = role_list + [PageRole.RECENT] * (cache_n - len(role_list))
        mask = plan_int4_mask(role_list, plan_cfg, policy=policy)
        packed, state = pack_prefill_to_fi_state(
            past,
            role_list,
            mask,
            device=device,
            dtype=torch.bfloat16,
            int4_cfg=int4_cfg,
            decode_tail_cap=max(decode_tail_cap, max_new_tokens + 8),
        )

        stub = FiSeqLenCache(state)
        attn = torch.ones(1, n, dtype=torch.long, device=device)
        gen_ids: list[int] = []
        with fi_shim_context(state) as ctx:
            # Replay last prompt token through FI path.
            replay = model(
                input_ids=ids[-1:].view(1, 1),
                attention_mask=attn,
                past_key_values=stub,
                use_cache=True,
                return_dict=True,
            )
            next_id = int(torch.argmax(replay.logits[:, -1, :], dim=-1).item())
            gen_ids.append(next_id)
            cur = torch.tensor([[next_id]], device=device)
            for _ in range(max_new_tokens - 1):
                attn = torch.cat(
                    [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)],
                    dim=1,
                )
                step = model(
                    input_ids=cur,
                    attention_mask=attn,
                    past_key_values=stub,
                    use_cache=True,
                    return_dict=True,
                )
                nid = int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
                gen_ids.append(nid)
                cur = torch.tensor([[nid]], device=device)
                if tokenizer.eos_token_id is not None and nid == tokenizer.eos_token_id:
                    break
            used_mat = bool(ctx.used_materialize)

        state.assert_no_materialize_path(used_mat)

    return {
        "token_ids": gen_ids,
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "used_materialize_hf_past": used_mat,
        "cache_tokens": cache_n,
        "prompt_tokens": n,
        "int4_tokens": int(mask.sum()),
        "payload_bytes": packed.payload_bytes(),
        "decode_len": int(state.decode_len),
        "attn_backend": "flashinfer_fi_shim",
    }


def greedy_materialize_baseline(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    *,
    roles: Sequence[PageRole],
    plan_cfg: MixedPlanConfig,
    policy: str = "structure",
    max_new_tokens: int = 8,
    int4_cfg: Optional[Int4KvConfig] = None,
) -> dict[str, Any]:
    """Same split-prefill contract via materialize→SDPA (oracle for token match)."""
    import torch
    from prioritykv.packed_mixed_cache import apply_packed_int4_to_hf_past

    if input_ids.dim() == 1:
        ids = input_ids
    else:
        ids = input_ids[0]
    n = int(ids.numel())
    device = model.device
    ids = ids.to(device)
    cache_n = n - 1
    int4_cfg = int4_cfg or Int4KvConfig()

    with torch.no_grad():
        pre = model(
            input_ids=ids[:cache_n].unsqueeze(0),
            attention_mask=torch.ones(1, cache_n, dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        past = pre.past_key_values
        role_list = list(roles)
        if len(role_list) != cache_n:
            if len(role_list) > cache_n:
                role_list = role_list[:cache_n]
            else:
                role_list = role_list + [PageRole.RECENT] * (cache_n - len(role_list))
        mask = plan_int4_mask(role_list, plan_cfg, policy=policy)
        past, _packed = apply_packed_int4_to_hf_past(
            past,
            role_list,
            mask,
            int4_cfg=int4_cfg,
            device=device,
            dtype=torch.bfloat16,
        )
        attn = torch.ones(1, n, dtype=torch.long, device=device)
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
        cur = torch.tensor([[next_id]], device=device)
        for _ in range(max_new_tokens - 1):
            attn = torch.cat(
                [attn, torch.ones((1, 1), device=device, dtype=attn.dtype)],
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
            cur = torch.tensor([[nid]], device=device)
            if tokenizer.eos_token_id is not None and nid == tokenizer.eos_token_id:
                break

    return {
        "token_ids": gen_ids,
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "used_materialize_hf_past": True,
        "attn_backend": "sdpa_materialize",
    }
