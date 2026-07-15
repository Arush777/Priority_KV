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
        "next": (
            "wire FlashInfer homogeneous paged attention + LSE merge on H200"
            if ok
            else "uv add flashinfer (H200 gpu extra) then re-check; CPU LSE ref already gates correctness"
        ),
    }


def try_import_flashinfer() -> Optional[Any]:
    if not flashinfer_available():
        return None
    import flashinfer

    return flashinfer
