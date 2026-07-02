"""
Patterns 2 & 5: Graph RAG + Knowledge Graph
=============================================
Uses a proper NetworkX bipartite graph:
  - Entity nodes :  "e::<entity_lower>"
  - Chunk  nodes :  "c::<chunk_id>"
  - Edges        :  chunk ↔ entity (when chunk mentions that entity)
  - Chunk–chunk  :  implicit via shared entity neighbours

Graph traversal (BFS, depth-2) finds chunks that are entity-connected to the
question entities but were NOT retrieved by vector search — the hallmark of
Graph RAG that pure vector search misses.

The graph is built in-memory during ingestion and rebuilt from Qdrant on
process restart via `rebuild_from_qdrant()`.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Global in-memory graph (thread-safe via lock)
# ---------------------------------------------------------------------------
_graph: nx.Graph = nx.Graph()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Node name helpers
# ---------------------------------------------------------------------------

def _entity_node(entity: str) -> str:
    return f"e::{entity.lower().strip()}"


def _chunk_node(chunk_id: str) -> str:
    return f"c::{chunk_id}"


# ---------------------------------------------------------------------------
# Ingestion — called by the arq worker after entity extraction
# ---------------------------------------------------------------------------

def add_chunk_entities(
    chunk_id: str, entities: List[str], collection_id: str
) -> None:
    """
    Register chunk in the knowledge graph:
      1. Add a chunk node (c::<chunk_id>) with metadata.
      2. For each entity add an entity node (e::<entity>) and connect it.
      3. Increment co-occurrence weights between chunk pairs that share entities.
    """
    c_node = _chunk_node(chunk_id)

    with _lock:
        # Chunk node
        _graph.add_node(c_node, kind="chunk", collection_id=collection_id, chunk_id=chunk_id)

        for entity in entities:
            e_node = _entity_node(entity)

            # Entity node
            _graph.add_node(e_node, kind="entity", label=entity)

            # Chunk ↔ Entity edge
            _graph.add_edge(c_node, e_node)

            # Strengthen chunk–chunk co-occurrence via this entity
            for neighbour in list(_graph.neighbors(e_node)):
                if neighbour == c_node:
                    continue
                if _graph.nodes[neighbour].get("kind") == "chunk":
                    # Add / increment a direct chunk–chunk edge weighted by shared entities
                    if _graph.has_edge(c_node, neighbour):
                        _graph[c_node][neighbour]["weight"] = (
                            _graph[c_node][neighbour].get("weight", 1) + 1
                        )
                    else:
                        _graph.add_edge(c_node, neighbour, weight=1)


# ---------------------------------------------------------------------------
# Query-time — called by Stage 3.5 of the RAG pipeline
# ---------------------------------------------------------------------------

def get_related_chunk_ids(
    entities: List[str],
    exclude_ids: Set[str],
    collection_ids: List[str],
    limit: int = 5,
    bfs_depth: int = 2,
) -> List[Tuple[str, str]]:
    """
    BFS traversal from each entity node (depth = `bfs_depth`) to find
    chunk nodes that:
      • are NOT already in `exclude_ids`
      • belong to one of `collection_ids`

    Returns up to `limit` (chunk_id, collection_id) pairs, sorted by
    descending direct edge-weight (co-occurrence strength).
    """
    allowed_cols = set(collection_ids)
    candidates: Dict[str, Tuple[str, str, int]] = {}  # chunk_id → (chunk_id, col_id, weight)

    with _lock:
        for entity in entities:
            e_node = _entity_node(entity)
            if e_node not in _graph:
                continue

            # BFS from the entity node — walks entity→chunk→entity→chunk paths
            bfs_tree = nx.bfs_tree(_graph, e_node, depth_limit=bfs_depth)
            for node in bfs_tree.nodes():
                if _graph.nodes.get(node, {}).get("kind") != "chunk":
                    continue
                chunk_id = _graph.nodes[node]["chunk_id"]
                col_id   = _graph.nodes[node].get("collection_id", "")
                if chunk_id in exclude_ids or col_id not in allowed_cols:
                    continue
                weight = _graph.edges.get((e_node, node), {}).get("weight", 1)
                if chunk_id not in candidates or candidates[chunk_id][2] < weight:
                    candidates[chunk_id] = (chunk_id, col_id, weight)

    # Sort by co-occurrence weight descending
    ranked = sorted(candidates.values(), key=lambda x: x[2], reverse=True)

    results: List[Tuple[str, str]] = []
    for chunk_id, col_id, _ in ranked[:limit]:
        exclude_ids.add(chunk_id)
        results.append((chunk_id, col_id))

    return results


# ---------------------------------------------------------------------------
# Graph introspection helpers (used by /graph/stats API endpoint)
# ---------------------------------------------------------------------------

def graph_stats() -> Dict:
    with _lock:
        chunk_nodes  = [n for n, d in _graph.nodes(data=True) if d.get("kind") == "chunk"]
        entity_nodes = [n for n, d in _graph.nodes(data=True) if d.get("kind") == "entity"]
        return {
            "total_nodes":   _graph.number_of_nodes(),
            "total_edges":   _graph.number_of_edges(),
            "chunk_nodes":   len(chunk_nodes),
            "entity_nodes":  len(entity_nodes),
            "top_entities":  _top_entities(10),
        }


def _top_entities(n: int = 10) -> List[Dict]:
    """Return the n entities with the most connected chunks."""
    entity_nodes = [
        (node, _graph.degree(node))
        for node, data in _graph.nodes(data=True)
        if data.get("kind") == "entity"
    ]
    entity_nodes.sort(key=lambda x: x[1], reverse=True)
    return [
        {"entity": _graph.nodes[node]["label"], "connections": deg}
        for node, deg in entity_nodes[:n]
    ]


# ---------------------------------------------------------------------------
# Rebuild graph from Qdrant on process restart
# ---------------------------------------------------------------------------

def rebuild_from_qdrant(collection_ids: List[str]) -> None:
    """
    Scroll through Qdrant collections and rebuild the in-memory graph.
    Called at API startup.  Silently skips collections that are unavailable.
    """
    try:
        from rag.vector_store import get_client
        client = get_client()
        for col_id in collection_ids:
            try:
                offset = None
                while True:
                    results, next_offset = client.scroll(
                        collection_name=col_id,
                        limit=200,
                        offset=offset,
                        with_payload=True,
                    )
                    for point in results:
                        p = point.payload or {}
                        if p.get("type") != "parent":
                            continue
                        entities = p.get("entities", [])
                        if entities:
                            add_chunk_entities(str(point.id), entities, col_id)
                    if next_offset is None:
                        break
                    offset = next_offset
                print(f"[graph] Rebuilt graph for collection '{col_id}'")
            except Exception as exc:
                print(f"[graph] Could not rebuild '{col_id}': {exc}")
    except Exception as exc:
        print(f"[graph] rebuild_from_qdrant failed: {exc}")