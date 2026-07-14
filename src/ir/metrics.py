"""Classic IR ranking metrics for a single query (or mean over queries).

Inputs use ranked doc id lists and a set (or graded dict) of relevant docs.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Set, Union

Relevance = Union[Set[str], Mapping[str, float]]


def _relevant_ids(qrels: Relevance) -> Set[str]:
    if isinstance(qrels, Mapping):
        return {doc_id for doc_id, grade in qrels.items() if grade > 0}
    return set(qrels)


def _grade(qrels: Relevance, doc_id: str) -> float:
    if isinstance(qrels, Mapping):
        return float(qrels.get(doc_id, 0.0))
    return 1.0 if doc_id in qrels else 0.0


def recall_at_k(ranked: Sequence[str], qrels: Relevance, k: int) -> float:
    """Fraction of relevant docs found in the top-k ranks."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _relevant_ids(qrels)
    if not rel:
        return 0.0
    hit = sum(1 for doc_id in ranked[:k] if doc_id in rel)
    return hit / len(rel)


def mrr(ranked: Sequence[str], qrels: Relevance) -> float:
    """Mean Reciprocal Rank for one query (reciprocal of first relevant rank)."""
    rel = _relevant_ids(qrels)
    if not rel:
        return 0.0
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], qrels: Relevance, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary or graded qrels)."""
    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(order: Sequence[str]) -> float:
        total = 0.0
        for i, doc_id in enumerate(order[:k], start=1):
            gain = _grade(qrels, doc_id)
            if gain <= 0:
                continue
            total += (2.0**gain - 1.0) / math.log2(i + 1)
        return total

    actual = dcg(ranked)
    if actual == 0.0:
        return 0.0
    # Ideal order: all positively graded docs, highest grade first
    if isinstance(qrels, Mapping):
        ideal_ids = sorted(
            (doc_id for doc_id, g in qrels.items() if g > 0),
            key=lambda d: qrels[d],
            reverse=True,
        )
    else:
        ideal_ids = sorted(qrels)
    ideal = dcg(ideal_ids)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mean_metric(
    ranked_by_query: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Relevance],
    metric: str,
    k: int | None = None,
) -> float:
    """Average a per-query metric over shared query ids."""
    qids = sorted(set(ranked_by_query) & set(qrels_by_query))
    if not qids:
        return 0.0
    scores: list[float] = []
    for qid in qids:
        ranked = ranked_by_query[qid]
        qrels = qrels_by_query[qid]
        if metric == "recall":
            if k is None:
                raise ValueError("k required for recall")
            scores.append(recall_at_k(ranked, qrels, k))
        elif metric == "mrr":
            scores.append(mrr(ranked, qrels))
        elif metric == "ndcg":
            if k is None:
                raise ValueError("k required for ndcg")
            scores.append(ndcg_at_k(ranked, qrels, k))
        else:
            raise ValueError(f"unknown metric: {metric}")
    return sum(scores) / len(scores)
