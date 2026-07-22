"""All retention arms as kvpress presses, so only the *policy* differs.

Why this exists
---------------
The repo's original token-gather path physically rewrites the prompt to the
retained tokens and regenerates. kvpress instead prefills the **full** prompt and
evicts KV entries afterwards. Those are different interventions: deleting text
the model never sees is strictly more destructive than dropping an
already-computed KV entry.

Comparing a token-gather ``structure`` against a kvpress ``snapkv`` therefore
measures the *mechanism*, not the retention policy — which is the actual research
question. Here every arm is a press over the same prefill, at the same
compression ratio, differing only in which KV entries it keeps:

``full``       no press
``structure``  role-scored :class:`StructureScorerPress` (this module)
``uniform``    ``StreamingLLMPress``  (attention sinks + recent window)
``random``     ``RandomPress``        (seeded, position-blind)
``snapkv``     ``SnapKVPress``        (real attention-based selection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from prioritykv.baselines.keep_policy import KeepPolicyConfig
from prioritykv.page_roles import PROTECTED_ROLES, PageRole

# Priority bands, highest kept first. Gaps are wide enough that a within-band
# positional tiebreak can never cross a boundary.
#
# Protected roles deliberately share ONE band. The frozen select_structure walks
# `sorted(struct)` in plain index order with no preference between CONSTRAINT,
# TOOL, SYSTEM or OTHER; ranking them separately would change *which* tokens the
# policy keeps, and the only thing this module is allowed to change is the
# mechanism by which they are dropped.
_BAND_SINK = 7.0e6
_BAND_RECENT = 6.0e6
_BAND_STRUCTURE = 5.0e6
_BAND_OTHER = 0.0

_STRUCTURE_ROLES = frozenset(PROTECTED_ROLES) | {PageRole.OTHER}


def structure_token_scores(
    n: int, roles: Sequence[PageRole], cfg: KeepPolicyConfig
) -> np.ndarray:
    """Per-token keep priority for the structure policy (higher = keep).

    Mirrors the frozen ``select_structure`` ordering — hard sink, forced recent
    window, then protected roles, then everything else preferring tokens nearer
    the decision point — but expressed as a score so a press can rank with it.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    scores = np.empty(n, dtype=np.float64)
    sink_end = min(cfg.sink_tokens, n)
    recent_start = max(0, n - cfg.force_recent)

    for i in range(n):
        role = roles[i] if i < len(roles) else PageRole.FILLER
        if i < sink_end:
            band, prefer_early = _BAND_SINK, True
        elif i >= recent_start:
            band, prefer_early = _BAND_RECENT, True
        elif role in _STRUCTURE_ROLES:
            # Frozen select_structure adds protected tokens in index order, so
            # when the budget cannot hold them all the *earliest* survive.
            band, prefer_early = _BAND_STRUCTURE, True
        else:
            # Frozen fills the remainder walking left from the recent edge, so
            # among unprotected tokens the *latest* survive.
            band, prefer_early = _BAND_OTHER, False

        frac = i / max(1, n)
        # Tiebreak stays strictly inside the band gap, so it can never reorder bands.
        scores[i] = band + (1.0 - frac if prefer_early else frac) * 1.0e5

    return scores


def make_structure_press(compression_ratio: float):
    """Construct a :class:`StructureScorerPress` (imports kvpress lazily)."""
    from kvpress import ScorerPress

    @dataclass
    class StructureScorerPress(ScorerPress):
        """Rank KV entries by application-visible structure rather than attention."""

        compression_ratio: float = 0.0
        token_scores: np.ndarray | None = field(default=None)

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            import torch

            bsz, n_kv_heads, k_len, _ = keys.shape
            if self.token_scores is None:
                raise RuntimeError("structure press used before token_scores was set")
            s = np.asarray(self.token_scores, dtype=np.float32)
            if s.shape[0] != k_len:
                raise RuntimeError(
                    f"structure score length {s.shape[0]} != KV length {k_len}; "
                    "role alignment is broken"
                )
            t = torch.as_tensor(s, device=keys.device, dtype=torch.float32)
            return t.view(1, 1, k_len).expand(bsz, n_kv_heads, k_len)

    return StructureScorerPress(compression_ratio=float(compression_ratio))


def make_uniform_press(compression_ratio: float, *, n_sink: int = 16):
    """Position-blind sink + recent window, the kvpress analogue of `uniform`."""
    from kvpress import StreamingLLMPress

    return StreamingLLMPress(compression_ratio=float(compression_ratio), n_sink=n_sink)


def make_random_press(compression_ratio: float, *, seed: int = 0):
    """Seeded position-blind control. Genuinely random, unlike the frozen core's.

    kvpress's own ``RandomPress(seed=...)`` builds a CPU ``torch.Generator`` and
    then samples against CUDA keys, which raises "Expected a 'cuda' device type
    for generator but found 'cpu'". Scores are drawn here with a NumPy RNG and
    moved to the keys' device instead, which is both device-agnostic and exactly
    reproducible from the seed.
    """
    from kvpress import ScorerPress

    @dataclass
    class SeededRandomPress(ScorerPress):
        """Position-blind random keep, reproducible and device-safe."""

        compression_ratio: float = 0.0
        seed: int = 0

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            import torch

            bsz, n_kv_heads, k_len, _ = keys.shape
            rng = np.random.default_rng(self.seed)
            s = rng.random(k_len, dtype=np.float32)
            t = torch.as_tensor(s, device=keys.device, dtype=torch.float32)
            return t.view(1, 1, k_len).expand(bsz, n_kv_heads, k_len)

    return SeededRandomPress(compression_ratio=float(compression_ratio), seed=int(seed))


def make_snapkv_press_ext(compression_ratio: float, *, window_size: int = 64,
                          kernel_size: int = 5):
    from kvpress import SnapKVPress

    return SnapKVPress(compression_ratio=float(compression_ratio),
                       window_size=window_size, kernel_size=kernel_size)


def compression_ratio_for_budget(n_tokens: int, budget: int) -> float:
    """Ratio that makes ``int(n * (1 - ratio))`` land on the shared budget."""
    if n_tokens <= 0:
        return 0.0
    return max(0.0, min(0.999, 1.0 - budget / n_tokens))


def expected_kept(n_tokens: int, compression_ratio: float) -> int:
    """kvpress keeps exactly this many entries (see ``ScorerPress.compress``)."""
    return int(n_tokens * (1 - compression_ratio))


PRESS_ARMS: tuple[str, ...] = ("structure", "uniform", "random", "snapkv")


def press_class_name(arm: str) -> str:
    return {
        "structure": "prioritykv.external.presses.StructureScorerPress",
        "uniform": "kvpress.StreamingLLMPress",
        "random": "kvpress.RandomPress",
        "snapkv": "kvpress.SnapKVPress",
    }[arm]
