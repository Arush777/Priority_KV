"""PriorityKV serving backend (page manager, INT4 path, mixed-precision attention)."""

__version__ = "0.1.0"

from prioritykv.byte_model import realized_bytes

__all__ = ["realized_bytes", "__version__"]
