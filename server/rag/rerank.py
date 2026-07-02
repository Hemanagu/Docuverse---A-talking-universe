"""
Stage 4.5: LLM Re-ranking
Ask the LLM to score each chunk 0–10 for relevance to the question.
Keep only the top-k chunks.
"""
import asyncio
import re
from typing import Any, Dict, List, Tuple

from rag.llm import llm_generate


async def _score_chunk(chunk: Dict[str, Any], question: str) -> Tuple[float, Dict[str, Any]]:
    prompt = (
        "Rate the relevance of the following text to the question on a scale "
        "from 0 to 10 (10 = perfectly answers the question, 0 = completely unrelated).\n"
        "Output ONLY a single integer score, nothing else.\n\n"
        f"Question: {question}\n\n"
        f"Text: {chunk['text'][:500]}\n\n"
        "Relevance score (0-10):"
    )
    raw = await llm_generate(prompt, max_tokens=10)
    # Extract the first integer 0–10 found in the response
    match = re.search(r"\b(10|[0-9])\b", raw)
    score = float(match.group(1)) if match else 5.0
    return score, chunk


async def rerank_chunks(
    chunks: List[Dict[str, Any]], question: str, top_k: int = 6
) -> List[Dict[str, Any]]:
    """
    Score all *chunks* concurrently then return the *top_k* highest-scored ones.
    """
    if not chunks:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = await asyncio.gather(
        *[_score_chunk(c, question) for c in chunks]
    )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]