"""
arq Worker — PDF ingestion pipeline
=====================================
Run with:
    python -m arq worker.WorkerSettings

Requires Redis / Valkey running on REDIS_HOST:REDIS_PORT.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from arq.connections import RedisSettings

from rag.chunker import parent_child_chunks
from rag.collections import update_collection_status
from rag.embedder import embed_texts
from rag.entity_extraction import extract_entities
from rag.knowledge_graph import add_chunk_entities
from rag.pdf_loader import extract_text_and_tables
from rag.vector_store import upsert_chunks


# ---------------------------------------------------------------------------
# Job handler
# ---------------------------------------------------------------------------

async def process_pdf_job(
    ctx,
    collection_id: str,
    pdf_path: str,
    filename: str,
) -> None:
    """
    Full ingestion pipeline executed by the arq worker:
      1. Parse PDF → pages (text + tables + image descriptions)
      2. parent_child_chunks() → 512-char parents + 128-char children
      3. Entity extraction (spaCy) on parent chunks → register in graph
      4. Embed all chunks with all-MiniLM-L6-v2
      5. Upsert chunks + vectors into Qdrant
      6. Mark collection status → "ready"
    """
    print(f"[worker] ▶ Processing '{filename}' → collection '{collection_id}'")

    try:
        # Step 1 – Parse PDF
        pages = await extract_text_and_tables(pdf_path)
        if not pages:
            print(f"[worker] ✗ No text extracted from '{filename}'")
            update_collection_status(collection_id, "error")
            return

        # Step 2 – Chunk
        all_chunks = []
        for page in pages:
            chunks = parent_child_chunks(
                page["text"],
                page_num=page["page"],
                source=filename,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            update_collection_status(collection_id, "error")
            return

        print(f"[worker]   {len(all_chunks)} chunks created "
              f"({sum(1 for c in all_chunks if c['type']=='parent')} parents, "
              f"{sum(1 for c in all_chunks if c['type']=='child')} children)")

        # Step 3 – Entity extraction + graph registration (parents only)
        parent_chunks = [c for c in all_chunks if c["type"] == "parent"]
        for chunk in parent_chunks:
            entities = extract_entities(chunk["text"])
            chunk["entities"] = entities
            add_chunk_entities(chunk["id"], entities, collection_id)

        # Step 4 – Embed all chunks
        texts = [c["text"] for c in all_chunks]
        vectors = embed_texts(texts)

        # Step 5 – Upsert into Qdrant
        upsert_chunks(collection_id, all_chunks, vectors)

        # Step 6 – Mark ready with chunk count
        total_chunks = len(all_chunks)
        update_collection_status(collection_id, "ready", chunk_count=total_chunks)
        print(f"[worker] ✓ Collection '{collection_id}' is READY ({total_chunks} chunks)")

    except Exception as exc:
        print(f"[worker] ✗ Error processing '{filename}': {exc}")
        update_collection_status(collection_id, "error")
        raise


# ---------------------------------------------------------------------------
# Worker settings
# ---------------------------------------------------------------------------

class WorkerSettings:
    functions = [process_pdf_job]
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
    )
    max_jobs = 4
    job_timeout = 600        # 10 min max per PDF
    keep_result = 3600       # keep job result for 1 hour