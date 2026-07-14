#!/usr/bin/env python3
"""Print SnapKV baseline scaffold status (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.baselines.snapkv import status  # noqa: E402

print(json.dumps(status(), separators=(",", ":")))
