"""Retention arms for the external BFCL evaluation, at a matched token budget.

Five primary arms:

``full``       FullKV control (no eviction).
``structure``  application-visible structure policy (protected roles + sink/recent).
``uniform``    position-blind matched keep.
``random``     deterministic seeded matched keep.
``snapkv``     *real* attention-based SnapKV via ``kvpress.SnapKVPress``.

Every non-``full`` arm is driven from one shared budget function so the realised
keep count is identical across arms for a given prefix. ``snapkv`` has no
fallback path: if ``kvpress`` is missing or its press does not actually compress
the cache, the run fails loudly rather than quietly degrading into DropKeep or
another heuristic wearing SnapKV's name.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from prioritybench.pins import chat_template_kwargs_for_tokenizer
from prioritykv.baselines.keep_policy import (
    KeepPolicyConfig,
    apply_keep_indices,
    assign_token_roles,
    select_keep_indices,
)

ARMS: tuple[str, ...] = ("full", "structure", "uniform", "snapkv", "random")
TOKEN_GATHER_ARMS = frozenset({"structure", "uniform", "random"})


class SnapKVUnavailableError(RuntimeError):
    """Raised when the real attention-based SnapKV path cannot run."""


def keep_budget(n_tokens: int, cfg: KeepPolicyConfig) -> int:
    """Shared matched-keep budget. Identical for every non-FullKV arm."""
    budget = max(cfg.sink_tokens + cfg.force_recent, int(round(n_tokens * cfg.keep_frac)))
    return min(budget, n_tokens)


def _chat_kwargs(tokenizer, enable_thinking: bool | None) -> dict:
    """Chat-template kwargs, with an explicit thinking override.

    The project default (prioritybench.pins) disables Qwen3 thinking so tool-call
    spans stay in the assistant channel the page tagger expects. That choice was
    made for the synthetic bench; on BFCL it materially changes model capability,
    so the external config states it explicitly rather than inheriting silently.
    """
    kwargs = dict(chat_template_kwargs_for_tokenizer(tokenizer))
    if enable_thinking is not None and "enable_thinking" in kwargs:
        kwargs["enable_thinking"] = bool(enable_thinking)
    return kwargs


def select_random_external(n: int, cfg: KeepPolicyConfig) -> np.ndarray:
    """A genuinely position-blind random keep, at the shared matched budget.

    The frozen core's ``select_random`` is byte-identical to ``select_uniform``
    at every context length: it sets ``recent = budget - sink_tokens``, so the
    forced sink+recent block already fills the whole budget, ``remaining_budget``
    is always 0, and the random branch never executes. That makes the published
    "random" baseline a duplicate of uniform rather than an independent control.

    This corrected version holds sink and recent at their *configured* sizes
    (16 and 128) and draws the remaining budget uniformly at random from the
    middle, which is what a position-blind control is supposed to be. It lives in
    the external namespace so nothing under FINAL_RUN_MANIFEST.yaml changes.
    """
    budget = keep_budget(n, cfg)
    if budget >= n:
        return np.arange(n, dtype=np.int64)

    sink = min(cfg.sink_tokens, n)
    recent = min(cfg.force_recent, max(0, n - sink))
    must = set(range(sink)) | set(range(n - recent, n))
    remaining = budget - len(must)

    if remaining > 0:
        middle = np.setdiff1d(np.arange(n, dtype=np.int64),
                              np.fromiter(must, dtype=np.int64, count=len(must)),
                              assume_unique=False)
        if middle.size:
            rng = np.random.default_rng(cfg.seed)
            pick = rng.choice(middle, size=min(remaining, middle.size), replace=False)
            must.update(int(x) for x in pick)

    return np.array(sorted(must), dtype=np.int64)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    requested_keep: int
    realized_keep: int
    kept_indices: np.ndarray | None
    timings: dict[str, float]
    extra: dict[str, Any]


class Generator(Protocol):
    arm: str

    def generate(self, messages: Sequence[dict], max_new_tokens: int) -> GenerationResult: ...


# --------------------------------------------------------------------------- #
# Token-gather arms (structure / uniform / random) and FullKV
# --------------------------------------------------------------------------- #


class TokenGatherGenerator:
    """FullKV, or matched-keep by gathering retained tokens into a dense prefix.

    This is the same RoPE-safe regenerate path the frozen PriorityBench-A runs
    use, so external numbers stay mechanically comparable to the core results.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        arm: str,
        keep_cfg: KeepPolicyConfig,
        max_model_len: int = 32768,
        enable_thinking: bool | None = None,
    ):
        if arm != "full" and arm not in TOKEN_GATHER_ARMS:
            raise ValueError(f"{arm!r} is not a token-gather arm")
        self.model = model
        self.tok = tokenizer
        self.arm = arm
        self.keep_cfg = keep_cfg
        self.max_model_len = max_model_len
        self.chat_kwargs = _chat_kwargs(tokenizer, enable_thinking)
        self._step = 0

    def _render(self, messages: Sequence[dict]) -> str:
        return self.tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **self.chat_kwargs
        )

    def generate(self, messages: Sequence[dict], max_new_tokens: int) -> GenerationResult:
        import torch

        t0 = time.perf_counter()
        text = self._render(messages)
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        n = int(ids.numel())

        kept_indices = None
        if self.arm == "full":
            kept_ids = ids
            requested = realized = n
        else:
            requested = keep_budget(n, self.keep_cfg)
            roles = None
            if self.arm == "structure":
                roles = assign_token_roles(
                    self.tok, list(messages), chat_kwargs=self.chat_kwargs
                )
                if len(roles) != n:
                    raise RuntimeError(
                        f"structure role/token misalignment: roles={len(roles)} n={n}"
                    )
            cfg = self.keep_cfg
            if self.arm == "random":
                # Decorrelate the mask across steps without losing determinism.
                cfg = KeepPolicyConfig(**{**self.keep_cfg.__dict__,
                                          "seed": self.keep_cfg.seed + self._step})
            if self.arm == "random":
                idx = select_random_external(n, cfg)
            else:
                idx = select_keep_indices(n, cfg, policy=self.arm, roles=roles)
            kept_ids = apply_keep_indices(ids, idx)
            kept_indices = idx
            realized = int(kept_ids.numel())

        t_sel = time.perf_counter()
        inputs = {
            "input_ids": kept_ids.unsqueeze(0).to(self.model.device),
            "attention_mask": torch.ones(
                1, kept_ids.numel(), dtype=torch.long, device=self.model.device
            ),
        }
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        new_ids = gen[0, inputs["input_ids"].shape[-1]:].tolist()
        out_text = self.tok.decode(new_ids, skip_special_tokens=True)
        t_end = time.perf_counter()
        self._step += 1

        return GenerationResult(
            text=out_text,
            prompt_tokens=n,
            requested_keep=requested,
            realized_keep=realized,
            kept_indices=kept_indices,
            timings={"select_s": t_sel - t0, "generate_s": t_end - t_sel,
                     "total_s": t_end - t0},
            extra={"granularity": self.keep_cfg.granularity, "new_tokens": len(new_ids)},
        )


# --------------------------------------------------------------------------- #
# Real attention-based SnapKV
# --------------------------------------------------------------------------- #


def make_snapkv_press(compression_ratio: float, *, window_size: int = 64,
                      kernel_size: int = 5):
    """Construct a genuine ``kvpress.SnapKVPress``. No substitutes."""
    try:
        from kvpress import SnapKVPress
    except Exception as exc:  # noqa: BLE001
        raise SnapKVUnavailableError(
            "kvpress is not importable; the snapkv arm must not fall back to "
            "DropKeep or any other heuristic"
        ) from exc
    return SnapKVPress(
        compression_ratio=float(compression_ratio),
        window_size=window_size,
        kernel_size=kernel_size,
    )


def assert_real_snapkv(press) -> None:
    """Fail unless ``press`` is genuinely ``kvpress.SnapKVPress``."""
    try:
        from kvpress import SnapKVPress
    except Exception as exc:  # noqa: BLE001
        raise SnapKVUnavailableError("kvpress unavailable") from exc
    if not isinstance(press, SnapKVPress):
        raise SnapKVUnavailableError(
            f"expected kvpress.SnapKVPress, got {type(press).__module__}."
            f"{type(press).__name__}"
        )


class SnapKVGenerator:
    """Matched-budget SnapKV using the real kvpress press hook.

    ``compression_ratio`` is recomputed per step from the shared
    :func:`keep_budget`, so SnapKV evicts to the same token count the
    token-gather arms retain instead of to a fixed fraction.
    """

    arm = "snapkv"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        keep_cfg: KeepPolicyConfig,
        window_size: int = 64,
        kernel_size: int = 5,
        max_model_len: int = 32768,
        enable_thinking: bool | None = None,
    ):
        self.model = model
        self.tok = tokenizer
        self.keep_cfg = keep_cfg
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.max_model_len = max_model_len
        self.chat_kwargs = _chat_kwargs(tokenizer, enable_thinking)
        # Probe once at construction so a missing/incorrect press fails before
        # any GPU budget is spent.
        assert_real_snapkv(make_snapkv_press(0.5, window_size=window_size,
                                            kernel_size=kernel_size))

    def generate(self, messages: Sequence[dict], max_new_tokens: int) -> GenerationResult:
        import torch

        t0 = time.perf_counter()
        text = self.tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **self.chat_kwargs
        )
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        n = int(ids.numel())
        requested = keep_budget(n, self.keep_cfg)
        ratio = 0.0 if n <= 0 else max(0.0, min(1.0, 1.0 - requested / n))

        press = make_snapkv_press(ratio, window_size=self.window_size,
                                  kernel_size=self.kernel_size)
        assert_real_snapkv(press)

        inputs = {
            "input_ids": ids.unsqueeze(0).to(self.model.device),
            "attention_mask": torch.ones(1, n, dtype=torch.long, device=self.model.device),
        }
        t_sel = time.perf_counter()
        with torch.no_grad(), press(self.model):
            # return_dict_in_generate is required to get past_key_values back;
            # without it generate() returns a bare tensor and the realised keep
            # count is unmeasurable, which would let the matched-budget check
            # pass vacuously for the one arm that most needs verifying.
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                return_dict_in_generate=True,
            )
        sequences = gen.sequences
        new_ids = sequences[0, n:].tolist()
        out_text = self.tok.decode(new_ids, skip_special_tokens=True)
        t_end = time.perf_counter()

        realized = measure_cache_length(gen, self.model, n, len(new_ids))
        if realized < 0:
            raise SnapKVUnavailableError(
                "could not measure the compressed KV length, so the SnapKV arm's "
                "realised keep count cannot be verified against the shared budget"
            )
        return GenerationResult(
            text=out_text,
            prompt_tokens=n,
            requested_keep=requested,
            realized_keep=realized,
            kept_indices=None,
            timings={"select_s": t_sel - t0, "generate_s": t_end - t_sel,
                     "total_s": t_end - t0},
            extra={
                "compression_ratio": ratio,
                "window_size": self.window_size,
                "kernel_size": self.kernel_size,
                "press_class": "kvpress.SnapKVPress",
                "new_tokens": len(new_ids),
            },
        )


def measure_cache_length(gen_out, model, prompt_tokens: int, new_tokens: int) -> int:
    """Realised prompt-KV length after a compressed prefill, or -1 if unknown.

    Returns the number of *prompt* KV entries the cache actually holds, so it can
    be compared against the shared keep budget.
    """
    cache = getattr(gen_out, "past_key_values", None)
    if cache is None and isinstance(gen_out, dict):
        cache = gen_out.get("past_key_values")
    if cache is None:
        return -1
    try:
        length = int(cache.get_seq_length())
    except Exception:  # noqa: BLE001
        try:
            length = int(cache[0][0].shape[-2])
        except Exception:  # noqa: BLE001
            return -1
    # The cache also holds the freshly decoded tokens.
    return max(0, length - new_tokens)


def check_matched_budget(
    per_arm_realized: dict[str, int],
    *,
    tolerance: int = 0,
) -> tuple[bool, str]:
    """Verify every non-FullKV arm realised the same keep count."""
    unknown = [a for a, v in per_arm_realized.items() if a != "full" and v < 0]
    if unknown:
        return False, f"realised keep count unknown for {unknown}; cannot verify match"
    vals = {a: v for a, v in per_arm_realized.items() if a != "full"}
    if len(vals) < 2:
        return True, "insufficient arms to compare"
    lo, hi = min(vals.values()), max(vals.values())
    if hi - lo > tolerance:
        return False, f"keep-count mismatch across arms (spread {hi - lo}): {vals}"
    return True, f"matched at {lo}"
