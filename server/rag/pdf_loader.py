"""
PDF Loader — pdfplumber + Y-coordinate table detection + Ollama vision image description.
"""
import io
import os
import base64
from typing import Any, Dict, List

import httpx
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-11b-vision-preview")


# ---------------------------------------------------------------------------
# Table detection via Y-coordinate clustering
# ---------------------------------------------------------------------------

def _cluster_words_by_y(words: List[Dict], tolerance: int = 5) -> List[List[Dict]]:
    """Group PDF word objects into rows by proximity on the Y axis."""
    rows: Dict[int, List[Dict]] = {}
    for word in words:
        y_bucket = round(word["top"] / tolerance) * tolerance
        rows.setdefault(y_bucket, []).append(word)
    return [
        sorted(row, key=lambda w: w["x0"])
        for row in sorted(rows.values(), key=lambda r: r[0]["top"])
    ]


def _detect_tables(page) -> List[str]:
    """
    Detect table-like regions using Y-coordinate row clustering.
    A run of rows where each row has ≥ 3 tokens is treated as a table.
    """
    words = page.extract_words()
    if not words:
        return []

    rows = _cluster_words_by_y(words)
    tables: List[str] = []
    table_rows: List[List[Dict]] = []

    def _flush_table(rows_buf: List[List[Dict]]) -> None:
        if len(rows_buf) >= 2:
            table_text = "\n".join(
                " | ".join(w["text"] for w in row) for row in rows_buf
            )
            tables.append(f"[TABLE]\n{table_text}\n[/TABLE]")

    for row in rows:
        if len(row) >= 3:
            table_rows.append(row)
        else:
            _flush_table(table_rows)
            table_rows = []

    _flush_table(table_rows)  # flush any trailing table
    return tables


# ---------------------------------------------------------------------------
# Image description via Ollama llama3.2-vision (optional)
# ---------------------------------------------------------------------------

async def _describe_image(image_bytes: bytes) -> str:
    """Ask OpenRouter vision model to describe an image. Returns '' on failure."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return ""

    try:
        image_b64 = base64.b64encode(image_bytes).decode()
        data_url = f"data:image/png;base64,{image_b64}"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Describe the content of this image factually and concisely. Focus on data, diagrams, charts, or text visible in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url": data_url }
                        }
                    ]
                }
            ],
            "max_tokens": 256,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(GROQ_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                print(f"[pdf_loader] Image processing rate limited or failed: {resp.status_code}")
    except Exception as exc:
        print(f"[pdf_loader] Image description skipped: {exc}")
    return ""


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

async def extract_text_and_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF and return a list of page dicts:
        { "page": int, "text": str }

    Text content per page is assembled from:
      1. Raw extracted text
      2. Tables detected via Y-coordinate clustering (formatted as pipe-tables)
      3. Image descriptions from Ollama llama3.2-vision (if available)
    """
    pages: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            parts: List[str] = []

            # 1. Main text
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())

            # 2. Tables via Y-coord clustering
            for table_str in _detect_tables(page):
                parts.append(table_str)

            # 3. Images → vision LLM description
            for img_meta in page.images:
                try:
                    x0 = img_meta["x0"]
                    top = img_meta["top"]
                    x1 = img_meta["x1"]
                    bottom = img_meta["bottom"]
                    cropped = page.crop((x0, top, x1, bottom))
                    buf = io.BytesIO()
                    cropped.to_image(resolution=150).save(buf, format="PNG")
                    description = await _describe_image(buf.getvalue())
                    if description:
                        parts.append(f"[IMAGE DESCRIPTION]: {description}")
                except Exception:
                    pass  # images are best-effort

            combined = "\n\n".join(parts)
            if combined.strip():
                pages.append({"page": i + 1, "text": combined})

    return pages