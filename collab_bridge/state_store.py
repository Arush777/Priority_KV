from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class StateStore:
    """Persistent bridge state under repo state/ (gitignored)."""

    def __init__(self, state_dir: Path, agent_id: str):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.path = self.state_dir / f"bridge_{agent_id}.json"
        self.lock_path = self.state_dir / f"bridge_{agent_id}.lock"
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "telegram_offset": None,
                "paused": False,
                "pause_reason": None,
                "last_tick_at": None,
                "last_agent_id": None,
                "ticks": 0,
                "history": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def telegram_offset(self) -> int | None:
        val = self._data.get("telegram_offset")
        return int(val) if val is not None else None

    @telegram_offset.setter
    def telegram_offset(self, value: int | None) -> None:
        self._data["telegram_offset"] = value
        self.save()

    @property
    def paused(self) -> bool:
        return bool(self._data.get("paused"))

    def pause(self, reason: str) -> None:
        self._data["paused"] = True
        self._data["pause_reason"] = reason
        self.save()

    def resume(self) -> None:
        self._data["paused"] = False
        self._data["pause_reason"] = None
        self.save()

    def record_tick(self, summary: dict[str, Any]) -> None:
        self._data["last_tick_at"] = time.time()
        self._data["ticks"] = int(self._data.get("ticks") or 0) + 1
        hist = list(self._data.get("history") or [])
        hist.append(summary)
        self._data["history"] = hist[-50:]
        self.save()

    def acquire_lock(self, stale_sec: int = 7200) -> bool:
        now = time.time()
        if self.lock_path.exists():
            try:
                meta = json.loads(self.lock_path.read_text(encoding="utf-8"))
                age = now - float(meta.get("ts", 0))
                if age < stale_sec:
                    return False
            except Exception:
                pass
        self.lock_path.write_text(
            json.dumps({"ts": now, "agent_id": self.agent_id}),
            encoding="utf-8",
        )
        return True

    def release_lock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()
