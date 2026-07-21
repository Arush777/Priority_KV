"""Atomic per-work-unit checkpointing for interruptible Slurm shards.

One file per work unit, never one large mutable results JSON. A point is
complete only if it lands as valid JSON carrying every required field, so a job
killed mid-write leaves a ``.tmp`` that the next run discards and retries rather
than a truncated file that resume would mistake for finished work.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REQUIRED_POINT_FIELDS: tuple[str, ...] = (
    "work_id",
    "freeze_id",
    "dataset_revision",
    "task_id",
    "category",
    "model_id",
    "model_revision",
    "arm",
    "keep_frac",
    "seed",
    "harness_revision",
    "terminal_status",
)


class ResultStore:
    """Layout under ``$PRAJNA_ROOT/results/external_bfcl_prajna_v1/``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = self.root / "manifest"
        self.points = self.root / "points"
        self.failures = self.root / "failures"
        self.shard_logs = self.root / "shard_logs"
        self.summaries = self.root / "summaries"

    def ensure(self) -> "ResultStore":
        for d in (self.manifest, self.points, self.failures, self.shard_logs, self.summaries):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def point_path(self, work_id: str) -> Path:
        return self.points / f"{work_id}.json"

    def failure_path(self, work_id: str) -> Path:
        return self.failures / f"{work_id}.json"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` durably: tmp file → flush → fsync → atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leave nothing half-written behind on interrupt.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Durably link the rename itself.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return path


def validate_point(payload: dict[str, Any]) -> tuple[bool, str]:
    missing = [f for f in REQUIRED_POINT_FIELDS if f not in payload]
    if missing:
        return False, f"missing fields: {missing}"
    return True, "ok"


def load_valid_point(path: str | Path) -> dict[str, Any] | None:
    """Return the point only if it parses *and* validates; else ``None``."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ok, _ = validate_point(payload)
    return payload if ok else None


def write_point(store: ResultStore, payload: dict[str, Any]) -> Path:
    ok, why = validate_point(payload)
    if not ok:
        raise ValueError(f"refusing to write invalid point: {why}")
    return atomic_write_json(store.point_path(payload["work_id"]), payload)


def write_failure(store: ResultStore, payload: dict[str, Any]) -> Path:
    """Failures are first-class records, never silently dropped."""
    return atomic_write_json(store.failure_path(payload["work_id"]), payload)


def completed_work_ids(store: ResultStore) -> set[str]:
    """Work IDs whose point files are present *and* valid."""
    done: set[str] = set()
    if not store.points.is_dir():
        return done
    for p in store.points.glob("*.json"):
        payload = load_valid_point(p)
        if payload is not None:
            done.add(payload["work_id"])
    return done


def pending_work_items(
    work_items: Iterable[dict[str, Any]], store: ResultStore
) -> list[dict[str, Any]]:
    """Skip only validated-complete points; incomplete/corrupt ones are retried."""
    done = completed_work_ids(store)
    return [w for w in work_items if w["work_id"] not in done]


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shard:
    index: int
    work_items: list[dict[str, Any]]


def build_shards(work_items: list[dict[str, Any]], shard_size: int) -> list[Shard]:
    """Contiguous shards of ~``shard_size`` units so the model loads once each."""
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    return [
        Shard(index=i // shard_size, work_items=work_items[i: i + shard_size])
        for i in range(0, len(work_items), shard_size)
    ]


def write_shard_status(store: ResultStore, shard_index: int, status: dict[str, Any]) -> Path:
    return atomic_write_json(store.shard_logs / f"shard_{shard_index:05d}.json", status)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Atomic JSONL write (same durability contract as points)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
