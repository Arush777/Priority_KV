#!/usr/bin/env python3
"""Print Q2 INT4 baseline status (quanto / cache availability)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.int4_kv import status  # noqa: E402


def main() -> int:
    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
