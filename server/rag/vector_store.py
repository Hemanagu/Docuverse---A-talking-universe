"""
Pattern 14: Vector DB Fallback — Qdrant → in-memory Numpy
===========================================================
All vector operations go through the VectorStoreRouter which:
  1. Tries Qdrant first (persistent, full-featured)
  2. Falls back to an in-memory Numpy index if Qdrant is unreachable
     (data lives only for the process lifetime; useful for dev/offline)

The fallback is transparent to the caller — same upsert / search / get API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Vector dimension — imported from embedder so it stays in sync
def _get_vector_size() -> int:
    try:
        from rag.embedder import VECTOR_DIM
        return VECTOR_DIM
    except Exception:
        return 384


# ═══════════════════════════════════════════════════════════════════════════
# Numpy in-memory backend (100% Mac/ARM64 compatible, no FAISS C++ issues)
# ═══════════════════════════════════════════════════════════════════════════

class _NumpyBackend:
    """
    Pure in-memory Numpy index used when Qdrant is unavailable.
    Stores both vectors and full payload metadata per collection.
    """

    def __init__(self) -> None:
        self._collections: Dict[str, Dict] = {}
        # { collection_id: { "vectors": np.ndarray, "chunks": List[Dict] } }

    def _get_or_create(self, collection_id: str):
        if collection_id not in self._collections:
            self._collections[collection_id] = {
                "vectors": np.empty((0, _get_vector_size()), dtype="float32"),
                "chunks": []
            }
        return self._collections[collection_id]

    def ensure_collection(self, collection_id: str) -> None:
        self._get_or_create(collection_id)

    def upsert(
        self, collection_id: str, chunks: List[Dict], vectors: List[List[float]]
    ) -> None:
        col = self._get_or_create(collection_id)
        new_vecs = np.array(vectors, dtype="float32")
        col["vectors"] = np.vstack([col["vectors"], new_vecs]) if col["vectors"].shape[0] > 0 else new_vecs
        col["chunks"].extend(chunks)
        print(f"[numpy-fallback] Upserted {len(chunks)} → '{collection_id}'")

    def search(
        self, collection_id: str, query_vector: List[float], top_k: int = 10
    ) -> List[Dict]:
        if collection_id not in self._collections:
            return []
        
        col = self._collections[collection_id]
        vecs = col["vectors"]
        chunks = col["chunks"]
        
        if vecs.shape[0] == 0:
            return []
            
        q_vec = np.array(query_vector, dtype="float32")
        # Dot product (inner product) for cosine similarity on normalized vectors
        scores = np.dot(vecs, q_vec)
        
        # Get top K indices
        k = min(top_k, len(scores))
        # argpartition is faster than full sort
        top_indices = np.argpartition(scores, -k)[-k:]
        # sort the top k by score descending
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for idx in top_indices:
            r = chunks[idx].copy()
            r["score"] = float(scores[idx])
            r.setdefault("_collection_id", collection_id)
            results.append(r)
            
        return results

    def get_by_id(self, collection_id: str, chunk_id: str) -> Optional[Dict]:
        if collection_id not in self._collections:
            return None
        for chunk in self._collections[collection_id]["chunks"]:
            if chunk.get("id") == chunk_id:
                return chunk.copy()
        return None

    def get_by_entities(
        self, collection_id: str, entities: List[str], limit: int = 5
    ) -> List[Dict]:
        if collection_id not in self._collections:
            return []
        results = []
        for chunk in self._collections[collection_id]["chunks"]:
            chunk_ents = {e.lower() for e in chunk.get("entities", [])}
            if any(e.lower() in chunk_ents for e in entities):
                results.append(chunk.copy())
                if len(results) >= limit:
                    break
        return results

    def delete_collection(self, collection_id: str) -> None:
        self._collections.pop(collection_id, None)


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant backend helpers
# ═══════════════════════════════════════════════════════════════════════════

_qdrant_client = None
_qdrant_available: Optional[bool] = None   # None = not yet tested


def _get_qdrant():
    global _qdrant_client, _qdrant_available
    if _qdrant_available is False:
        return None
    try:
        if _qdrant_client is None:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=5)
        # Quick health check
        _qdrant_client.get_collections()
        _qdrant_available = True
        return _qdrant_client
    except Exception as exc:
        if _qdrant_available is not False:
            print(f"[vector_store] Qdrant unavailable: {exc}. Switching to Numpy fallback.")
        _qdrant_available = False
        return None


def _ensure_qdrant_collection(client, collection_id: str) -> None:
    from qdrant_client.models import Distance, VectorParams
    existing = {c.name for c in client.get_collections().collections}
    if collection_id not in existing:
        client.create_collection(
            collection_name=collection_id,
            vectors_config=VectorParams(size=_get_vector_size(), distance=Distance.COSINE),
        )


def _hit_to_dict(hit, collection_id: str) -> Dict[str, Any]:
    p = hit.payload or {}
    return {
        "id": str(hit.id),
        "score": getattr(hit, "score", 0.0),
        "text": p.get("text", ""),
        "type": p.get("type", "child"),
        "chunk_type": p.get("chunk_type", "text"),   # text | table | image
        "page": p.get("page", 1),
        "source": p.get("source", ""),
        "parent_id": p.get("parent_id"),
        "entities": p.get("entities", []),
        "_collection_id": p.get("collection_id", collection_id),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Shared Numpy instance (module-level singleton)
# ═══════════════════════════════════════════════════════════════════════════

_numpy_backend = _NumpyBackend()


# ═══════════════════════════════════════════════════════════════════════════
# Public API — same for both backends
# ═══════════════════════════════════════════════════════════════════════════

def get_client():
    """Returns Qdrant client or None (for code that needs raw client)."""
    return _get_qdrant()


def upsert_chunks(
    collection_id: str, chunks: List[Dict[str, Any]], vectors: List[List[float]]
) -> None:
    client = _get_qdrant()
    if client:
        try:
            from qdrant_client.models import PointStruct
            _ensure_qdrant_collection(client, collection_id)
            points = [
                PointStruct(
                    id=chunk["id"],
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "type": chunk.get("type", "parent"),
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "page": chunk.get("page", 1),
                        "source": chunk.get("source", ""),
                        "parent_id": chunk.get("parent_id"),
                        "entities": chunk.get("entities", []),
                        "collection_id": collection_id,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            batch_size = 256
            for i in range(0, len(points), batch_size):
                client.upsert(collection_name=collection_id, points=points[i : i + batch_size])
            print(f"[qdrant] Upserted {len(points)} points → '{collection_id}'")
            return
        except Exception as exc:
            print(f"[qdrant] Upsert failed: {exc}. Using Numpy fallback.")

    # Numpy fallback
    _numpy_backend.upsert(collection_id, chunks, vectors)


def search_collection(
    collection_id: str, query_vector: List[float], top_k: int = 10
) -> List[Dict[str, Any]]:
    client = _get_qdrant()
    if client:
        try:
            # qdrant-client >= 1.7: use query_points (client.search was removed)
            result = client.query_points(
                collection_name=collection_id,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = result.points
            return [_hit_to_dict(h, collection_id) for h in hits]
        except AttributeError:
            # Fallback for qdrant-client < 1.7 that still has search()
            try:
                hits = client.search(  # type: ignore[attr-defined]
                    collection_name=collection_id,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                return [_hit_to_dict(h, collection_id) for h in hits]
            except Exception as exc:
                print(f"[qdrant] Search (legacy) failed: {exc}. Using Numpy fallback.")
        except Exception as exc:
            print(f"[qdrant] Search failed: {exc}. Using Numpy fallback.")

    return _numpy_backend.search(collection_id, query_vector, top_k)


def get_chunk_by_id(
    collection_id: str, chunk_id: str
) -> Optional[Dict[str, Any]]:
    client = _get_qdrant()
    if client:
        try:
            results = client.retrieve(
                collection_name=collection_id, ids=[chunk_id], with_payload=True
            )
            if results:
                return _hit_to_dict(results[0], collection_id)
        except Exception as exc:
            print(f"[qdrant] Retrieve failed: {exc}. Trying Numpy.")

    return _numpy_backend.get_by_id(collection_id, chunk_id)


def get_chunks_by_entities(
    collection_id: str, entities: List[str], limit: int = 5
) -> List[Dict[str, Any]]:
    client = _get_qdrant()
    if client:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny
            results, _ = client.scroll(
                collection_name=collection_id,
                scroll_filter=Filter(
                    must=[FieldCondition(key="entities", match=MatchAny(any=entities))]
                ),
                limit=limit,
                with_payload=True,
            )
            return [_hit_to_dict(p, collection_id) for p in results]
        except Exception as exc:
            print(f"[qdrant] Entity scroll failed: {exc}. Trying Numpy.")

    return _numpy_backend.get_by_entities(collection_id, entities, limit)


def delete_collection_data(collection_id: str) -> None:
    client = _get_qdrant()
    if client:
        try:
            client.delete_collection(collection_id)
        except Exception:
            pass
    _numpy_backend.delete_collection(collection_id)


def backend_status() -> Dict[str, Any]:
    """Return current vector backend status."""
    client = _get_qdrant()
    if client:
        try:
            cols = [c.name for c in client.get_collections().collections]
            return {"backend": "qdrant", "url": QDRANT_URL, "collections": cols}
        except Exception:
            pass
    cols = list(_numpy_backend._collections.keys())
    return {"backend": "numpy_memory", "collections": cols}
