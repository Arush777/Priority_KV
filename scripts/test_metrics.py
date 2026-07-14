"""Unit tests for IR ranking metrics (no third-party deps)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir.metrics import mean_metric, mrr, ndcg_at_k, recall_at_k  # noqa: E402


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    qrels = {"a", "c", "z"}
    assert recall_at_k(ranked, qrels, k=2) == 1 / 3
    assert recall_at_k(ranked, qrels, k=3) == 2 / 3


def test_mrr():
    assert mrr(["x", "y", "rel"], {"rel"}) == 1 / 3
    assert mrr(["miss"], {"rel"}) == 0.0


def test_ndcg_binary():
    ranked = ["rel", "noise", "rel2"]
    qrels = {"rel", "rel2"}
    score = ndcg_at_k(ranked, qrels, k=3)
    assert 0.0 < score <= 1.0


def test_mean_metric():
    ranked = {"q1": ["a", "b"], "q2": ["c"]}
    qrels = {"q1": {"a"}, "q2": {"c"}}
    assert mean_metric(ranked, qrels, "mrr") == 1.0
    assert mean_metric(ranked, qrels, "recall", k=1) == 1.0


if __name__ == "__main__":
    test_recall_at_k()
    test_mrr()
    test_ndcg_binary()
    test_mean_metric()
    print("ok")
