"""FlashInfer multi-call hook (W4→W6).

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


def try_import_flashinfer() -> Optional[Any]:
    if not flashinfer_available():
        return None
    import flashinfer

    return flashinfer


def probe() -> dict[str, Any]:
    """Import + tiny CUDA smoke if package present; never silently claim CUDA."""
    fi = try_import_flashinfer()
    if fi is None:
        return {
            "available": False,
            "decision": "SKIP_NO_PACKAGE",
            "note": (
                "flashinfer not installed; CPU lse_merge_pair / "
                "mixed_attend_kv_multicall remain the correctness oracle. "
                "Install only via uv extra on H200 when wiring kernels."
            ),
        }
    out: dict[str, Any] = {
        "available": True,
        "package": getattr(fi, "__file__", str(fi)),
        "version": getattr(fi, "__version__", None),
    }
    try:
        import torch

        if not torch.cuda.is_available():
            out["decision"] = "IMPORT_OK_NO_CUDA"
            out["note"] = "flashinfer importable but CUDA unavailable in this process"
            return out
        # Minimal device touch — not a full multicall kernel yet.
        x = torch.zeros(1, device="cuda")
        out["cuda_device"] = torch.cuda.get_device_name(0)
        out["cuda_touch_ok"] = bool(x.numel() == 1)
        out["decision"] = "IMPORT_OK_CUDA_TOUCH"
        out["note"] = (
            "Package present; homogeneous paged multicall + LSE merge not wired yet. "
            "Next: implement CUDA path to match mixed_attend_kv_multicall."
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["decision"] = "IMPORT_OK_CUDA_FAIL"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def status() -> dict[str, Any]:
    ok = flashinfer_available()
    return {
        "name": "flashinfer_multicall",
        "available": ok,
        "lse_merge": "prioritykv.mixed_cache_reference.lse_merge_pair",
        "parity_oracle": "mixed_attend_kv_multicall vs mixed_attend_kv",
        "decision": "W6_PROBE",
        "next": (
            "implement FlashInfer homogeneous paged attention + LSE merge on H200"
            if ok
            else "uv sync --extra flashinfer (when added) on H200; CPU LSE oracle stays gating"
        ),
    }
