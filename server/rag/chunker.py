"""
Parent-Child Chunker with content-type awareness
  • Parent chunks : 512 characters  (semantic context window)
  • Child  chunks : 128 characters  (fine-grained retrieval units)

Recognises special content markers from pdf_loader:
  [TABLE] ... [/TABLE]  → chunk_type = "table"
  [IMAGE DESCRIPTION]   → chunk_type = "image"
  plain text            → chunk_type = "text"

Each child carries a `parent_id` so we can swap it back for its parent
during the Parent-Expansion stage of the RAG pipeline.
"""
import re
import uuid
from typing import Any, Dict, List


def _detect_content_type(text: str) -> str:
    """Detect if a text block is a table, image description, or plain text."""
    t = text.strip()
    if t.startswith("[TABLE]") or "[TABLE]" in t:
        return "table"
    if t.startswith("[IMAGE DESCRIPTION]"):
        return "image"
    return "text"


def parent_child_chunks(
    text: str,
    page_num: int = 1,
    source: str = "",
    parent_size: int = 512,
    child_size: int = 128,
) -> List[Dict[str, Any]]:
    """
    Split *text* into overlapping parent chunks and further subdivide each
    parent into child chunks, preserving table/image content types.

    Returns a flat list of chunk dicts with keys:
        id, text, type ("parent"|"child"), chunk_type ("text"|"table"|"image"),
        page, source, parent_id, entities
    """
    chunks: List[Dict[str, Any]] = []

    if not text or not text.strip():
        return chunks

    # First split on special block markers so tables/images stay intact
    # We split on [TABLE]...[/TABLE] and [IMAGE DESCRIPTION]:... blocks
    block_pattern = re.compile(
        r'(\[TABLE\].*?\[/TABLE\]|\[IMAGE DESCRIPTION\]:.*?)(?=\[TABLE\]|\[IMAGE DESCRIPTION\]|$)',
        re.DOTALL,
    )

    # If no special blocks found, treat whole text as plain text
    special_blocks = block_pattern.findall(text)

    # Build list of (text_segment, content_type) to chunk
    if not special_blocks:
        segments = [(text, "text")]
    else:
        segments = []
        remaining = text
        for block in special_blocks:
            block = block.strip()
            if not block:
                continue
            idx = remaining.find(block)
            if idx > 0:
                plain = remaining[:idx].strip()
                if plain:
                    segments.append((plain, "text"))
            segments.append((block, _detect_content_type(block)))
            remaining = remaining[idx + len(block):]
        if remaining.strip():
            segments.append((remaining.strip(), "text"))

    for seg_text, chunk_type in segments:
        if not seg_text.strip():
            continue

        # Slide over the segment in parent_size steps
        for p_start in range(0, len(seg_text), parent_size):
            parent_text = seg_text[p_start : p_start + parent_size].strip()
            if not parent_text:
                continue

            parent_id = str(uuid.uuid4())
            parent_chunk: Dict[str, Any] = {
                "id": parent_id,
                "text": parent_text,
                "type": "parent",
                "chunk_type": chunk_type,
                "page": page_num,
                "source": source,
                "parent_id": None,
                "entities": [],
            }
            chunks.append(parent_chunk)

            # Subdivide the parent into child chunks
            for c_start in range(0, len(parent_text), child_size):
                child_text = parent_text[c_start : c_start + child_size].strip()
                if not child_text:
                    continue
                child_chunk: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    "text": child_text,
                    "type": "child",
                    "chunk_type": chunk_type,
                    "page": page_num,
                    "source": source,
                    "parent_id": parent_id,
                    "entities": [],
                }
                chunks.append(child_chunk)

    return chunks