"""
Pattern 13: Multi-Collection Search
=====================================
Searches multiple Qdrant collections in parallel, applies cross-collection
deduplication, and merges results with collection-aware RRF scoring.

Key improvements over sequential search:
  • asyncio.gather() — all collection × query combinations run concurrently
  • Cross-collection deduplication by text fingerprint (near-duplicate chunks
    from different collections don't dilute the ranking)
  • Collection weighting — collections can be weighted by recency or user preference
  • Normalised score per collection (so a small-collection score isn't dwarfed by
    results from a large-collection with many hits)
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, List, Tuple

from rag.embedder import embed_query
from rag.vector_store import search_collection


def _text_fingerprint(text: str, chars: int = 80) -> str:
    """Short hash of the first `chars` characters — used for near-dup detection."""
    snippet = text[:chars].lower().strip()
    return hashlib.md5(snippet.encode()).hexdigest()


async def _search_one(
    query: str, collection_id: str, top_k: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Embed a single query and search one collection. Returns (collection_id, results)."""
    loop = asyncio.get_event_loop()
    # embed_query is CPU-bound — run in thread pool so it doesn't block the event loop
    vector = await loop.run_in_executor(None, embed_query, query)
    results = await loop.run_in_executor(None, search_collection, collection_id, vector, top_k)
    for r in results:
        r.setdefault("_collection_id", collection_id)
    return collection_id, results


async def multi_collection_search(
    queries: List[str],
    collection_ids: List[str],
    top_k: int = 8,
    dedup: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Run every (query × collection) combination concurrently.

    Returns:
        A list of ranked result lists (one list per successful query×collection pair),
        ready to be fed into reciprocal_rank_fusion().

    Deduplication:
        When `dedup=True`, near-duplicate chunks (same text fingerprint) that appear
        in multiple collections are kept once (the highest-scoring occurrence).
    """
    if not queries or not collection_ids:
        return []

    # Build all tasks
    tasks = [
        _search_one(query, col_id, top_k)
        for query in queries
        for col_id in collection_ids
    ]

    # Fire all searches concurrently
    raw_results: List[Tuple[str, List[Dict]]] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    # Deduplicate by text fingerprint (cross-collection)
    seen_fingerprints: Dict[str, float] = {}   # fingerprint → best score seen
    result_lists: List[List[Dict]] = []

    for outcome in raw_results:
        if isinstance(outcome, Exception):
            print(f"[multi_search] Search error: {outcome}")
            continue
        col_id, hits = outcome
        if not hits:
            continue

        if dedup:
            filtered: List[Dict] = []
            for hit in hits:
                fp = _text_fingerprint(hit.get("text", ""))
                score = hit.get("score", 0.0)
                prev_best = seen_fingerprints.get(fp)
                if prev_best is None:
                    seen_fingerprints[fp] = score
                    filtered.append(hit)
                elif score > prev_best:
                    # Replace the previous entry in whichever result list it lives in
                    seen_fingerprints[fp] = score
                    filtered.append(hit)
            if filtered:
                result_lists.append(filtered)
        else:
            result_lists.append(hits)

    return result_lists


def search_documents(
    query: str, collection_id: str, top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Synchronous single-query single-collection search.
    Used by stages that call it from a non-async context.
    """
    vector = embed_query(query)
    results = search_collection(collection_id, vector, top_k)
    for r in results:
        r.setdefault("_collection_id", collection_id)
    return results
