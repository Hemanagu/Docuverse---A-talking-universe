"""
Pattern 12 + 14: Embedding Pipeline with Fallback Strategy
===========================================================
Embedding model cascade:
  Primary  : all-MiniLM-L6-v2   (384-dim, cosine) — fast & free
  Fallback1: paraphrase-MiniLM-L3-v2 (384-dim)    — lighter model
  Fallback2: TF-IDF + SVD (BM25-style keywords)   — zero neural deps

The active model is determined once at startup.  If the primary model fails
(e.g., OOM or download error), the next in the chain is tried automatically.

VECTOR_DIM is published so that Qdrant collection creation can use the right
dimension regardless of which model is active.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Model cascade definition
# ---------------------------------------------------------------------------

_EMBEDDING_CHAIN: List[str] = [
    os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "sentence-transformers/paraphrase-MiniLM-L3-v2",   # fallback 1
    # "__tfidf__" is handled separately below               fallback 2
]

# Public — updated once the active model is determined
VECTOR_DIM: int = 384

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_st_model = None          # SentenceTransformer instance or None
_tfidf_vectorizer = None  # fitted TfidfVectorizer
_svd = None               # fitted TruncatedSVD for dimensionality reduction
_active_backend: str = "none"  # "sentence_transformer" | "tfidf" | "none"


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------

def _try_load_sentence_transformer() -> bool:
    global _st_model, VECTOR_DIM, _active_backend
    for model_name in _EMBEDDING_CHAIN:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[embedder] Loading '{model_name}' …")
            _st_model = SentenceTransformer(model_name)
            # Warm up to get actual output dimension
            sample = _st_model.encode(["warmup"], normalize_embeddings=True)
            VECTOR_DIM = sample.shape[1]
            _active_backend = "sentence_transformer"
            print(f"[embedder] ✓ Using '{model_name}' (dim={VECTOR_DIM})")
            return True
        except Exception as exc:
            print(f"[embedder] '{model_name}' failed: {exc}. Trying next …")
    return False


def _init_tfidf_backend(corpus: Optional[List[str]] = None) -> None:
    """
    TF-IDF + TruncatedSVD fallback.
    Produces 384-dim dense vectors (LSA-style) from keyword frequencies.
    The vectorizer is fitted lazily on the first batch of texts.
    """
    global _tfidf_vectorizer, _svd, VECTOR_DIM, _active_backend
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    _svd = TruncatedSVD(n_components=384, random_state=42)
    VECTOR_DIM = 384
    _active_backend = "tfidf"

    if corpus:
        tfidf_matrix = _tfidf_vectorizer.fit_transform(corpus)
        _svd.fit(tfidf_matrix)
        print(f"[embedder] TF-IDF+SVD fitted on {len(corpus)} texts")
    else:
        print("[embedder] TF-IDF+SVD backend ready (not yet fitted)")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _ensure_initialized(corpus: Optional[List[str]] = None) -> None:
    global _active_backend
    if _active_backend != "none":
        return

    if _try_load_sentence_transformer():
        return

    # sentence-transformers unavailable — fall back to TF-IDF
    print("[embedder] Falling back to TF-IDF + SVD (no neural model).")
    _init_tfidf_backend(corpus)


# ---------------------------------------------------------------------------
# TF-IDF encoding helpers
# ---------------------------------------------------------------------------

def _tfidf_encode(texts: List[str]) -> np.ndarray:
    global _tfidf_vectorizer, _svd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    if _tfidf_vectorizer is None:
        _tfidf_vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1, 2), sublinear_tf=True)
        _svd = TruncatedSVD(n_components=384, random_state=42)

    try:
        matrix = _tfidf_vectorizer.transform(texts)
        vecs = _svd.transform(matrix)
    except Exception:
        # Vectorizer not yet fitted — fit + transform
        matrix = _tfidf_vectorizer.fit_transform(texts)
        vecs = _svd.fit_transform(matrix)

    # L2-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of strings.
    Returns a list of VECTOR_DIM-dimensional normalized float vectors.
    Also triggers TF-IDF fitting if that backend is active and unfitted.
    """
    _ensure_initialized(corpus=texts)

    if _active_backend == "sentence_transformer" and _st_model is not None:
        embs = _st_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return embs.tolist()

    if _active_backend == "tfidf":
        return _tfidf_encode(texts).tolist()

    # Should never reach here
    dim = VECTOR_DIM
    return [[0.0] * dim for _ in texts]


def embed_query(query: str) -> List[float]:
    """Embed a single query string using the active backend."""
    return embed_texts([query])[0]


def active_backend_info() -> dict:
    """Return info about the currently active embedding backend."""
    return {
        "backend": _active_backend,
        "vector_dim": VECTOR_DIM,
        "model": (
            getattr(_st_model, "_model_card_data", {}).get("model_name", "sentence_transformer")
            if _st_model else ("tfidf+svd" if _active_backend == "tfidf" else "none")
        ),
    }