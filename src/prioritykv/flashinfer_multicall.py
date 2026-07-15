"""FlashInfer multi-call hook (W4).

Exact LSE merge lives in ``mixed_cache_reference``. CUDA FlashInfer path is
optional — loud-skip when the package is absent so CPU CI stays green.
"""

from __future__ import annotations

from typing import Any, Optional


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except Exception:
        return False


def status() -> dict[str, Any]:
    ok = flashinfer_available()
    return {
        "name": "flashinfer_multicall",
        "available": ok,
        "lse_merge": "prioritykv.mixed_cache_reference.lse_merge_pair",
        "parity_oracle": "mixed_attend_kv_multicall vs mixed_attend_kv",
        "decision": "DEFERRED_W5_W6",
        "next": (
            "optional: wire FlashInfer homogeneous paged attention + LSE merge on H200 "
            "(CPU lse_merge_pair / mixed_attend_kv_multicall already gate correctness)"
            if ok
            else "W4 closed with loud-skip; CPU LSE parity is the correctness oracle until W5–6"
        ),
    }


def try_import_flashinfer() -> Optional[Any]:
    if not flashinfer_available():
        return None
    import flashinfer

    return flashinfer
