"""
Stage 1: Query Expansion
LLM generates 3 semantically diverse query variants to improve recall.
"""
from typing import List

from rag.llm import llm_generate


async def expand_query(question: str) -> List[str]:
    """
    Return the original question + up to 3 LLM-generated variants.
    Each variant approaches the topic from a different angle.
    """
    prompt = (
        "Generate exactly 3 different search queries to retrieve information "
        "about the question below. Each query should approach the topic from a "
        "different angle (e.g., definition, cause/effect, examples).\n"
        "Output ONLY the 3 queries, one per line, with no numbering or preamble.\n\n"
        f"Original question: {question}\n\n"
        "3 queries:"
    )

    raw = await llm_generate(prompt, max_tokens=200)
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    variants = lines[:3]

    # Pad if LLM returned fewer than 3 lines
    fallbacks = [
        f"Explain {question}",
        f"What is the definition of {question}",
        f"Provide details about {question}",
    ]
    while len(variants) < 3:
        variants.append(fallbacks[len(variants)])

    return [question] + variants  # original + 3 variants → 4 total