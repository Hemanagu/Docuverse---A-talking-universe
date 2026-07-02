"""
Stage 0: HyDE (Hypothetical Document Embeddings)
Generate a plausible answer paragraph → embed it for retrieval.
"""
from rag.llm import llm_generate


async def generate_hypothetical_doc(question: str) -> str:
    """
    Ask the LLM to write a hypothetical answer paragraph.
    Embedding this answer often retrieves better passage matches than
    embedding the raw question.
    """
    prompt = (
        "Write a concise, factual paragraph that directly answers the following "
        "question. Write it as if you found this answer in a reference document. "
        "Use clear, documentary language.\n\n"
        f"Question: {question}\n\n"
        "Hypothetical answer paragraph:"
    )
    return await llm_generate(prompt, max_tokens=256)