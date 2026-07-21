"""BFCL V3 multi-turn dataset normalisation for EXTERNAL_BFCL_PRAJNA_V1.

Source of record is the *pinned Gorilla checkout*, not the Hugging Face mirror:
the mirror is stale (23 of 200 ``base`` questions differ) and its function docs
disagree with the API classes the official checker executes. Data, checker, and
stateful API implementations must come from one commit or scores are not
comparable to the leaderboard.

BFCL V3 multi-turn has exactly four categories, 200 conversations each. There is
no ``composite`` category in V3 (it arrives with V4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

CATEGORIES: tuple[str, ...] = ("base", "miss_param", "miss_func", "long_context")

# Categories whose scenarios are loaded with the long-context inflater.
LONG_CONTEXT_CATEGORIES = frozenset({"long_context", "composite"})


@dataclass(frozen=True)
class BfclTask:
    """One BFCL multi-turn conversation, normalised and ready to render."""

    task_id: str
    category: str
    question: list[list[dict[str, str]]]
    initial_config: dict[str, Any]
    involved_classes: list[str]
    ground_truth: list[list[str]]
    function: list[dict[str, Any]]
    missed_function: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.question)

    def as_test_entry(self) -> dict[str, Any]:
        """The dict shape the official ``multi_turn_checker`` expects."""
        return {
            "id": self.task_id,
            "question": [list(t) for t in self.question],
            "initial_config": self.initial_config,
            "involved_classes": list(self.involved_classes),
            "function": list(self.function),
            "missed_function": self.missed_function,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def data_dir(gorilla_root: str | Path) -> Path:
    return Path(gorilla_root) / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"


def _attach_function_docs(
    entry: dict[str, Any],
    func_doc_dir: Path,
    doc_mapping: dict[str, str],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Port of the official ``process_multi_turn_test_case`` doc attachment.

    Mirrors upstream exactly, including the miss-func holdout: the withheld
    function's doc is *removed* from the visible tool list and stashed under the
    turn index at which it is later revealed.
    """
    functions: list[dict[str, Any]] = []
    for class_name in entry["involved_classes"]:
        # The func_doc files are JSONL (one schema per line), matching how the
        # official loader reads them, not a single JSON array.
        functions.extend(_read_jsonl(func_doc_dir / doc_mapping[class_name]))

    missed: dict[str, list[dict[str, Any]]] = {}
    for turn_index, names in (entry.get("missed_function") or {}).items():
        held: list[dict[str, Any]] = []
        for name in names:
            for i, doc in enumerate(functions):
                if doc["name"] == name:
                    held.append(doc)
                    functions.pop(i)
                    break
        missed[str(turn_index)] = held
    return functions, missed


def load_tasks(
    gorilla_root: str | Path,
    *,
    categories: Sequence[str] = CATEGORIES,
    doc_mapping: dict[str, str] | None = None,
) -> list[BfclTask]:
    """Load and normalise the pinned multi-turn split."""
    root = data_dir(gorilla_root)
    func_doc_dir = root / "multi_turn_func_doc"
    if doc_mapping is None:
        from prioritykv.external.bfcl_official import load_official

        doc_mapping = load_official(str(gorilla_root))["MULTI_TURN_FUNC_DOC_FILE_MAPPING"]

    tasks: list[BfclTask] = []
    for category in categories:
        q_path = root / f"BFCL_v3_multi_turn_{category}.json"
        a_path = root / "possible_answer" / f"BFCL_v3_multi_turn_{category}.json"
        if not q_path.exists():
            raise FileNotFoundError(
                f"category {category!r} absent from pinned BFCL V3 data at {q_path}"
            )
        answers = {r["id"]: r["ground_truth"] for r in _read_jsonl(a_path)}
        for entry in _read_jsonl(q_path):
            functions, missed = _attach_function_docs(entry, func_doc_dir, doc_mapping)
            tasks.append(
                BfclTask(
                    task_id=entry["id"],
                    category=category,
                    question=[list(turn) for turn in entry["question"]],
                    initial_config=entry["initial_config"],
                    involved_classes=list(entry["involved_classes"]),
                    ground_truth=answers[entry["id"]],
                    function=functions,
                    missed_function=missed,
                )
            )
    return tasks


def build_system_prompt(task: BfclTask, default_system_prompt: str) -> str:
    """Official prompting-mode system prompt with this task's tool docs."""
    return default_system_prompt.format(functions=json.dumps(task.function))


def initial_messages(task: BfclTask, default_system_prompt: str) -> list[dict[str, str]]:
    """System turn only; user turns are appended by the rollout as it advances."""
    return [{"role": "system", "content": build_system_prompt(task, default_system_prompt)}]


# --------------------------------------------------------------------------- #
# Stable work identity
# --------------------------------------------------------------------------- #


def work_id(
    *,
    dataset_revision: str,
    task_id: str,
    model_revision: str,
    arm: str,
    keep_frac: float,
    seed: int,
    harness_revision: str,
    decision_turn: int | str = "all",
) -> str:
    """SHA-256 over exactly the fields that define one unit of work.

    ``decision_turn`` is ``"all"`` for the full-rollout protocol: the official
    multi-turn checker emits a single verdict per conversation, so the
    conversation is the unit and turns are never independent samples.
    """
    payload = "|".join(
        str(x)
        for x in (
            dataset_revision,
            task_id,
            decision_turn,
            model_revision,
            arm,
            f"{float(keep_frac):.6f}",
            seed,
            harness_revision,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def balanced_sample(
    tasks: Iterable[BfclTask],
    *,
    per_category: dict[str, int],
    seed: int = 0,
) -> list[BfclTask]:
    """Deterministic per-category sample, independent of input ordering.

    Selection ranks tasks by ``sha256(seed|task_id)`` rather than an RNG stream,
    so the chosen set for ``n`` is a prefix-stable function of the seed and does
    not shift when the sample size or category list changes.
    """
    by_cat: dict[str, list[BfclTask]] = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t)

    out: list[BfclTask] = []
    for category, n in sorted(per_category.items()):
        pool = by_cat.get(category, [])
        if n > len(pool):
            raise ValueError(
                f"requested {n} tasks for category {category!r} but only {len(pool)} exist"
            )
        ranked = sorted(
            pool,
            key=lambda t: hashlib.sha256(f"{seed}|{t.task_id}".encode()).hexdigest(),
        )
        out.extend(ranked[:n])
    return sorted(out, key=lambda t: (t.category, t.task_id))


def file_hashes(gorilla_root: str | Path, categories: Sequence[str] = CATEGORIES) -> dict:
    """SHA-256 of every data file the manifest depends on."""
    root = data_dir(gorilla_root)
    paths = []
    for category in categories:
        paths.append(root / f"BFCL_v3_multi_turn_{category}.json")
        paths.append(root / "possible_answer" / f"BFCL_v3_multi_turn_{category}.json")
    paths.extend(sorted((root / "multi_turn_func_doc").glob("*.json")))

    hashes = {}
    for p in paths:
        if p.exists():
            hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes
