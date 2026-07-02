"""
Stage 4: Parent Expansion
Swap child chunks for their richer parent chunks.
Each child carries a `parent_id` that points to the 512-char parent stored in Qdrant.
"""
from typing import Any, Dict, List

from rag.vector_store import get_chunk_by_id


def expand_to_parent_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For every child chunk, retrieve and substitute its parent chunk.
    Parent chunks (and already-parent chunks) are kept as-is.
    De-duplicates by parent ID so the same parent is not added twice.
    """
    expanded: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for chunk in chunks:
        chunk_type = chunk.get("type", "parent")
        parent_id = chunk.get("parent_id")
        col_id = chunk.get("_collection_id", "")

        if chunk_type == "child" and parent_id and col_id:
            if parent_id in seen_ids:
                continue  # already added this parent
            parent = get_chunk_by_id(col_id, parent_id)
            if parent:
                parent["_collection_id"] = col_id
                seen_ids.add(parent_id)
                expanded.append(parent)
                continue  # successfully replaced

        # Keep as-is (parent chunk or lookup failed)
        chunk_id = chunk.get("id", "")
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            expanded.append(chunk)

    return expanded