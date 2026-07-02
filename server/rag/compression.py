"""
Stage 5: Context Compression
LLM extracts only the sentences relevant to the question from each chunk.
"""
import asyncio
from typing import Any, Dict, List

from rag.llm import llm_generate


async def _compress_chunk(chunk: Dict[str, Any], question: str) -> str:
    prompt = (
        "From the text below, extract ONLY the sentences that directly help answer "
        "the question. If no sentences are relevant, reply with exactly: NOT_RELEVANT\n\n"
        f"Question: {question}\n\n"
        f"Text:\n{chunk['text']}\n\n"
        "Relevant sentences:"
    )
    result = await llm_generate(prompt, max_tokens=300)
    cleaned = result.strip()
    if not cleaned or cleaned.upper() == "NOT_RELEVANT":
        return ""
    return cleaned


async def compress_context(
    chunks: List[Dict[str, Any]], question: str
) -> List[str]:
    """
    Compress all *chunks* concurrently.
    Returns non-empty compressed strings only.
    """
    results: List[str] = await asyncio.gather(
        *[_compress_chunk(c, question) for c in chunks]
    )
    return [r for r in results if r]