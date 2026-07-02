"""
Summarizer — generate a 3-5 sentence summary for an uploaded collection.
"""
from rag.llm import llm_generate
from rag.vector_store import get_client


async def summarize_collection(collection_id: str) -> str:
    """Fetch the first N parent chunks and summarize with Ollama."""
    client = get_client()

    try:
        points, _ = client.scroll(
            collection_name=collection_id,
            limit=12,
            with_payload=True,
        )
    except Exception as exc:
        return f"Could not read collection: {exc}"

    parent_texts = [
        p.payload.get("text", "")
        for p in points
        if (p.payload or {}).get("type") == "parent"
    ]

    if not parent_texts:
        return "No content found in this collection."

    combined = "\n\n".join(parent_texts[:8])[:3500]

    prompt = (
        "Summarize the following document content in 3 to 5 concise sentences. "
        "Focus on the main topics, key findings, and important details.\n\n"
        f"{combined}\n\nSummary:"
    )

    return await llm_generate(prompt, max_tokens=400)