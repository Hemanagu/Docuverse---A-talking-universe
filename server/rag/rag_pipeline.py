"""
Full 7-Stage RAG Pipeline (v2)
=================================
Patterns implemented:
  1  Retrieval-Augmented Generation   — full pipeline
  2  Graph RAG                        — Stage 3.5 NetworkX BFS traversal
  3  Hierarchical chunking            — parent/child chunks (512/128)
  4  Vector similarity search         — Qdrant cosine + FAISS fallback
  5  Knowledge graphs                 — NetworkX entity graph
  6  Hypothetical embeddings (HyDE)   — Stage 0
  7  Query expansion                  — Stage 1  (LLM 3 variants)
  8  Reciprocal Rank Fusion           — Stage 3
  9  LLM re-ranking                   — Stage 4.5 (0-10 score)
 10  Context compression              — Stage 5
 11  Async job queues                 — arq + Valkey worker
 12  Embedding pipelines              — all-MiniLM → paraphrase → TF-IDF
 13  Multi-collection search          — concurrent asyncio.gather
 14  Model fallback strategies        — LLM chain + embedding chain + FAISS fallback

Pipeline flow:
  Stage 0   HyDE      → hypothetical answer → embed
  Stage 1   Expand    → LLM 3 variants
  Stage 2   Search    → multi-collection concurrent dense search
  Stage 3   RRF       → fuse/deduplicate, keep top 8
  Stage 3.5 Graph     → NetworkX BFS entity-connected neighbours
  Stage 4   Parent    → swap child chunks for 512-char parents
  Stage 4.5 Rerank    → LLM score 0-10, keep top 6
  Stage 5   Compress  → LLM extract relevant sentences
  Stage 6   Answer    → LLM cited answer + source cards
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from rag.compression import compress_context
from rag.entity_extraction import extract_entities
from rag.hyde import generate_hypothetical_doc
from rag.knowledge_graph import get_related_chunk_ids
from rag.llm import llm_generate
from rag.multi_search import multi_collection_search
from rag.parent_expansion import expand_to_parent_chunks
from rag.query_expansion import expand_query
from rag.rerank import rerank_chunks
from rag.rrf import reciprocal_rank_fusion
from rag.vector_store import get_chunk_by_id


async def run_rag_pipeline(
    question: str,
    collection_ids: List[str],
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    End-to-end 7-stage RAG pipeline.

    Returns:
        {
          "message": "<answer>",
          "docs":    [{ source, page, excerpt }, …],
          "meta":    { stages timing, backend info, … }
        }
    """
    t0 = time.perf_counter()

    if not collection_ids:
        return {
            "message": "Please upload a PDF first and wait for processing to complete.",
            "docs": [],
            "meta": {},
        }

    meta: Dict[str, Any] = {"collection_ids": collection_ids}

    # ── Stage 0: HyDE ──────────────────────────────────────────────────────
    t_stage = time.perf_counter()
    hyde_doc = await generate_hypothetical_doc(question)
    meta["stage0_hyde_ms"] = round((time.perf_counter() - t_stage) * 1000)

    # ── Stage 1: Query Expansion ───────────────────────────────────────────
    t_stage = time.perf_counter()
    expanded_queries = await expand_query(question)
    all_queries = expanded_queries + [hyde_doc]   # original + 3 LLM variants + HyDE
    meta["stage1_queries"] = len(all_queries)
    meta["stage1_expand_ms"] = round((time.perf_counter() - t_stage) * 1000)

    # ── Stage 2: Multi-Collection Dense Search (concurrent) ────────────────
    t_stage = time.perf_counter()
    retrieval_results = await multi_collection_search(
        queries=all_queries,
        collection_ids=collection_ids,
        top_k=8,
        dedup=True,   # cross-collection near-dup removal
    )
    meta["stage2_result_lists"] = len(retrieval_results)
    meta["stage2_search_ms"] = round((time.perf_counter() - t_stage) * 1000)

    if not retrieval_results:
        return {
            "message": "No relevant documents found. Try rephrasing your question.",
            "docs": [],
            "meta": meta,
        }

    # ── Stage 3: RRF Merge ─────────────────────────────────────────────────
    t_stage = time.perf_counter()
    ranked_docs = reciprocal_rank_fusion(retrieval_results)
    best_chunks = ranked_docs[:8]
    meta["stage3_rrf_ms"] = round((time.perf_counter() - t_stage) * 1000)
    meta["stage3_chunks"] = len(best_chunks)

    # ── Stage 3.5: Graph Expansion (NetworkX BFS) ──────────────────────────
    t_stage = time.perf_counter()
    question_entities = extract_entities(question)
    existing_ids = {c["id"] for c in best_chunks}

    graph_neighbors = get_related_chunk_ids(
        entities=question_entities,
        exclude_ids=existing_ids,
        collection_ids=collection_ids,
        limit=5,
        bfs_depth=2,
    )

    # Fetch graph-neighbour chunks from vector store (concurrent)
    async def _fetch_chunk(chunk_id: str, col_id: str) -> Dict | None:
        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(None, get_chunk_by_id, col_id, chunk_id)
        if chunk:
            chunk["_collection_id"] = col_id
            chunk["_from_graph"] = True
        return chunk

    graph_chunks = await asyncio.gather(
        *[_fetch_chunk(cid, col_id) for cid, col_id in graph_neighbors]
    )
    for gc in graph_chunks:
        if gc:
            best_chunks.append(gc)

    meta["stage3_5_graph_ms"] = round((time.perf_counter() - t_stage) * 1000)
    meta["stage3_5_graph_neighbors"] = len([g for g in graph_chunks if g])
    meta["stage3_5_entities"] = question_entities

    # ── Stage 4: Parent Expansion ──────────────────────────────────────────
    t_stage = time.perf_counter()
    parent_chunks = expand_to_parent_chunks(best_chunks)
    meta["stage4_parent_ms"] = round((time.perf_counter() - t_stage) * 1000)
    meta["stage4_chunks"] = len(parent_chunks)

    # ── Stage 4.5: LLM Re-rank ─────────────────────────────────────────────
    t_stage = time.perf_counter()
    reranked = await rerank_chunks(parent_chunks, question, top_k=6)
    meta["stage4_5_rerank_ms"] = round((time.perf_counter() - t_stage) * 1000)

    # ── Stage 5: Context Compression ──────────────────────────────────────
    t_stage = time.perf_counter()
    compressed = await compress_context(reranked, question)
    meta["stage5_compress_ms"] = round((time.perf_counter() - t_stage) * 1000)
    meta["stage5_compressed_blocks"] = len(compressed)

    # ── Stage 6: Answer Generation ────────────────────────────────────────
    t_stage = time.perf_counter()

    context_text = "\n\n---\n\n".join(compressed[:6]) if compressed else (
        "\n\n---\n\n".join(c["text"] for c in reranked[:4])
    )

    # Conversation history — handle both {role,content} (OpenAI fmt) and {user,assistant} (legacy)
    history_lines = []
    for turn in history[-12:]:            # keep last 6 exchanges (12 messages)
        if "role" in turn:                # OpenAI format: {role, content}
            role = turn["role"].capitalize()
            content = turn.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")
        else:                             # Legacy format: {user: ..., assistant: ...}
            if "user" in turn:
                history_lines.append(f"User: {turn['user']}")
            if "assistant" in turn:
                history_lines.append(f"Assistant: {turn['assistant']}")
    history_block = "\n".join(history_lines)

    answer_prompt = (
        "You are a precise document assistant. "
        "Answer ONLY using the provided context. "
        "Be concise and accurate. Cite the source document and page when possible.\n\n"
        + (f"Previous conversation:\n{history_block}\n\n" if history_block else "")
        + f"Context from documents:\n{context_text}\n\n"
        + f"Question: {question}\n\nAnswer:"
    )

    answer = await llm_generate(answer_prompt, max_tokens=600)
    meta["stage6_answer_ms"] = round((time.perf_counter() - t_stage) * 1000)

    # ── Build source citation cards ────────────────────────────────────────
    docs: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for chunk in reranked:
        key = f"{chunk.get('source', '')}::{chunk.get('page', 1)}"
        if key not in seen_keys:
            seen_keys.add(key)
            docs.append({
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", 1),
                "excerpt": chunk["text"][:220] + ("…" if len(chunk["text"]) > 220 else ""),
                "chunk_type": chunk.get("chunk_type", "text"),  # text | table | image
                "from_graph": chunk.get("_from_graph", False),
            })

    meta["total_ms"] = round((time.perf_counter() - t0) * 1000)

    return {"message": answer, "docs": docs, "meta": meta}