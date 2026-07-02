"""
Pattern 14: Model Fallback Strategies — OpenRouter LLM Fallback Chain
======================================================================
Instead of local Ollama (which requires ~7 GB of downloads), this uses
OpenRouter's free tier, greatly reducing local resource burden.

Cascading fallback chain for text generation:
  Primary  :  LLM_MODEL env var (default: meta-llama/llama-3.1-8b-instruct:free)
  Fallback 1:  google/gemini-2.0-flash-lite-preview-02-05:free
  Fallback 2:  mistralai/mistral-7b-instruct:free
  Last resort: Template-based rule extraction (no LLM, keyword grep)

The chain:
  1. Tries OpenRouter with the primary model.
  2. If it fails (e.g., rate limits), tries Google/Mistral free tiers.
  3. Falls back to a clean context excerpt if all APIs fail.

Both async and sync variants are provided.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ordered fallback chain of free models (updated for 2025-2026 availability)
# Primary model is from LLM_MODEL env var (so you can override easily)
_FALLBACK_CHAIN: List[str] = [
    os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

_DEFAULT_OPTIONS = {
    "temperature": 0.1,
    "max_tokens": 512,
}


# ---------------------------------------------------------------------------
# Template fallback (zero LLM, pure text)
# ---------------------------------------------------------------------------

def _template_answer(prompt: str) -> str:
    """
    Extractive QA fallback used only when ALL LLM APIs are unreachable.
    Finds the most relevant sentences from the document context by keyword
    overlap with the question, and formats a clean readable answer.
    """
    import re

    # Extract context block
    ctx_match = re.search(r"Context from documents:\n(.+?)\n\nQuestion:", prompt, re.DOTALL)
    question_match = re.search(r"Question:\s*(.+)", prompt)

    if not ctx_match:
        return (
            "⚠ The LLM API is temporarily unavailable (rate limited or quota exceeded). "
            "Please check your OPENROUTER_API_KEY. The document was indexed successfully — "
            "try again in a moment."
        )

    raw_context = ctx_match.group(1).strip()
    question = question_match.group(1).strip().lower() if question_match else ""

    # Split into sentences / meaningful lines
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', raw_context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Score sentences by keyword overlap with the question
    q_words = set(re.findall(r'\b\w{3,}\b', question)) - {
        'the', 'and', 'for', 'are', 'what', 'which', 'how', 'many', 'much',
        'does', 'can', 'this', 'that', 'with', 'have', 'from', 'about'
    }

    def score(s: str) -> int:
        s_low = s.lower()
        return sum(1 for w in q_words if w in s_low)

    scored = sorted(sentences, key=score, reverse=True)
    best = [s for s in scored[:6] if score(s) > 0] or scored[:4]

    if not best:
        return "No relevant content found for your question in the indexed documents."

    answer_blocks = []
    for s in best:
        if s.startswith('[TABLE]'):
            answer_blocks.append(s)
        elif s.startswith('[IMAGE DESCRIPTION]'):
            answer_blocks.append(s)
        else:
            answer_blocks.append(s)

    joined = "\n\n".join(answer_blocks)
    return (
        f"📄 **Extractive Answer** *(LLM API is rate-limited — showing relevant document excerpts)*\n\n"
        f"{joined}\n\n"
        f"*For a synthesized AI answer, wait a moment and try again.*"
    )


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------

async def llm_generate(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    """
    Try each model in the fallback chain via OpenRouter API until one succeeds.
    Falls back to template extraction if all fail.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[llm] GROQ_API_KEY is missing or strictly placeholder.")
        return _template_answer(prompt)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in _FALLBACK_CHAIN:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": _DEFAULT_OPTIONS["temperature"],
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()

                # OpenRouter returns standard OpenAI schema
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if text:
                    if model != os.getenv("LLM_MODEL"):
                        print(f"[llm] Rate limit trigger. Using fallback text model: {model}")
                    return text
        except Exception as exc:
            # Could be a 429 Rate Limit from Free Tier. Fallback to next model gracefully.
            err_msg = str(exc)
            if "429" in err_msg or "RateLimitError" in err_msg:
                print(f"[llm] Rate limited on '{model}'. Trying next model in chain...")
            else:
                print(f"[llm] Model '{model}' failed: {type(exc).__name__}: {exc}")
            continue

    # All models failed
    print("[llm] All models in fallback chain failed — using template extractor.")
    return _template_answer(prompt)


# ---------------------------------------------------------------------------
# Sync API (worker / blocking contexts)
# ---------------------------------------------------------------------------

def llm_generate_sync(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    """Synchronous version with the same fallback chain."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[llm-sync] GROQ_API_KEY is missing.")
        return _template_answer(prompt)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in _FALLBACK_CHAIN:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": _DEFAULT_OPTIONS["temperature"],
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()

                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if text:
                    return text
        except Exception as exc:
            print(f"[llm-sync] Model '{model}' failed: {exc}")
            continue

    return _template_answer(prompt)