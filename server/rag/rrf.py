"""
Stage 3: Reciprocal Rank Fusion (RRF)
Merges multiple ranked result lists into a single fused ranking.
"""
from typing import Any, Dict, List


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]], k: int = 60
) -> List[Dict[str, Any]]:
    """
    RRF score formula:  score(d) = Σ  1 / (k + rank(d))

    Args:
        results_list : list of ranked lists (each item is a chunk dict with `id`)
        k            : smoothing constant (default 60)

    Returns:
        Deduplicated list sorted by descending RRF score.
    """
    scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_store[doc_id] for doc_id, _ in ranked]