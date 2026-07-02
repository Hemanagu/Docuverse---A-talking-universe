"""
DocuVerse v2 — FastAPI Application
=====================================
Patterns 1-14 fully implemented.

Endpoints:
  GET  /                         → Premium chat UI
  GET  /status                   → Backend health (LLM, Qdrant, Redis, embedder)
  GET  /graph/stats              → Knowledge graph statistics
  POST /upload/pdf               → Upload PDF → queue ingestion job → {collectionId, status}
  GET  /collections              → List all collections with status
  GET  /collections/{id}/status  → Poll a single collection's status
  GET  /chat                     → 7-stage RAG answer
  GET  /summarize/{id}           → Summarize a collection
  DELETE /collections/{id}       → Delete collection
  GET  /audio/tts                → Text-to-Speech (returns MP3 blob)
  POST /audio/stt                → Speech-to-Text (accepts audio blob, returns JSON text)
"""
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------------------------------------------------------------------
# arq pool — gracefully degrade if Redis is unavailable
# ---------------------------------------------------------------------------
_arq_pool = None


async def _get_arq_pool():
    global _arq_pool
    if _arq_pool is None:
        try:
            import arq
            from arq.connections import RedisSettings

            _arq_pool = await arq.create_pool(
                RedisSettings(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                )
            )
        except Exception as exc:
            print(f"[app] Redis unavailable, will process PDFs inline: {exc}")
    return _arq_pool


# ---------------------------------------------------------------------------
# Fallback: process PDF inline when Redis is not available
# ---------------------------------------------------------------------------
async def _process_inline(collection_id: str, pdf_path: str, filename: str):
    from rag.chunker import parent_child_chunks
    from rag.collections import update_collection_status
    from rag.embedder import embed_texts
    from rag.entity_extraction import extract_entities
    from rag.knowledge_graph import add_chunk_entities
    from rag.pdf_loader import extract_text_and_tables
    from rag.vector_store import upsert_chunks

    try:
        pages = await extract_text_and_tables(pdf_path)
        all_chunks = []
        for page in pages:
            all_chunks.extend(
                parent_child_chunks(page["text"], page_num=page["page"], source=filename)
            )
        for chunk in [c for c in all_chunks if c["type"] == "parent"]:
            ents = extract_entities(chunk["text"])
            chunk["entities"] = ents
            add_chunk_entities(chunk["id"], ents, collection_id)
        vectors = embed_texts([c["text"] for c in all_chunks])
        upsert_chunks(collection_id, all_chunks, vectors)
        update_collection_status(collection_id, "ready")
        print(f"[app] Inline processing done → collection '{collection_id}'")
    except Exception as exc:
        from rag.collections import update_collection_status
        update_collection_status(collection_id, "error")
        print(f"[app] Inline processing error: {exc}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up Redis connection
    await _get_arq_pool()

    # Rebuild knowledge graph from existing Qdrant collections (Pattern 2 & 5)
    try:
        from rag.collections import read_collections
        from rag.knowledge_graph import rebuild_from_qdrant
        ready_cols = [c["id"] for c in read_collections() if c.get("status") == "ready"]
        if ready_cols:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, rebuild_from_qdrant, ready_cols)
    except Exception as exc:
        print(f"[app] Graph rebuild skipped: {exc}")

    yield
    if _arq_pool:
        await _arq_pool.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="DocuVerse", version="2.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js frontend (port 3000) to reach the API
# ---------------------------------------------------------------------------
_allowed_origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    displayName: str = Form(""),
):
    """
    Receive a PDF, save it, enqueue an arq job (or fall back to inline),
    and return { collectionId, status: "processing" }.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    collection_id = str(uuid.uuid4())[:8]
    safe_name = file.filename.replace(" ", "_")
    stored_filename = f"{collection_id}_{safe_name}"
    path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    from rag.collections import add_collection
    display = displayName.strip() or file.filename
    add_collection(collection_id, file.filename, display, "processing")

    pool = await _get_arq_pool()
    if pool:
        await pool.enqueue_job("process_pdf_job", collection_id, path, file.filename)
    else:
        # No Redis → process asynchronously in a background task
        import asyncio
        asyncio.create_task(_process_inline(collection_id, path, file.filename))

    return {"collectionId": collection_id, "status": "processing"}


@app.get("/collections")
async def list_collections():
    from rag.collections import read_collections
    return read_collections()


@app.get("/collections/{collection_id}/status")
async def collection_status(collection_id: str):
    from rag.collections import get_collection
    col = get_collection(collection_id)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return {"status": col.get("status", "unknown")}


@app.get("/chat")
async def chat(
    message: str,
    collections: str = "",
    history: str = "[]",
):
    """
    Run the 7-stage RAG pipeline.
    - collections: comma-separated collection IDs
    - history: JSON array of {user, assistant} dicts
    """
    from rag.rag_pipeline import run_rag_pipeline

    col_ids = [c.strip() for c in collections.split(",") if c.strip()]
    try:
        chat_history = json.loads(history)
    except json.JSONDecodeError:
        chat_history = []

    result = await run_rag_pipeline(message, col_ids, chat_history)
    return result


@app.get("/summarize/{collection_id}")
async def summarize(collection_id: str):
    from rag.summarizer import summarize_collection
    summary = await summarize_collection(collection_id)
    return {"summary": summary}


@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    from rag.collections import remove_collection
    from rag.vector_store import delete_collection_data
    delete_collection_data(collection_id)
    remove_collection(collection_id)
    return {"status": "deleted"}


@app.get("/audio/tts")
async def tts_endpoint(text: str):
    from rag.audio import text_to_speech
    audio_bytes = await text_to_speech(text)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty text or TTS failed")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/audio/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    from rag.audio import speech_to_text
    audio_bytes = await audio.read()
    
    # get extension to help pydub parse it properly (e.g. webm from browser)
    ext = audio.filename.split('.')[-1] if '.' in audio.filename else "webm"
    text = speech_to_text(audio_bytes, ext)
    
    return {"text": text}


@app.get("/graph/stats")
async def graph_stats():
    """Pattern 2 & 5: Return knowledge graph statistics."""
    from rag.knowledge_graph import graph_stats as _stats
    return _stats()


@app.get("/status")
async def system_status():
    """Pattern 14: Show all backend health statuses."""
    import httpx
    from rag.embedder import active_backend_info
    from rag.vector_store import backend_status

    # Check Ollama
    ollama_ok = False
    available_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags")
            if r.status_code == 200:
                ollama_ok = True
                available_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    # Check Redis/Valkey
    queue_ok = _arq_pool is not None

    return {
        "llm":       {"ok": ollama_ok, "models": available_models},
        "queue":     {"ok": queue_ok, "backend": "valkey/redis"},
        "vector_db": backend_status(),
        "embedder":  active_backend_info(),
    }