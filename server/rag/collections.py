"""
Collection registry — tracks collection IDs, display names, and statuses.
Backed by a JSON file so it survives process restarts.
"""
import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional

_FILE = os.path.join(os.path.dirname(__file__), "..", "collections.json")
_lock = Lock()


def _abs_path() -> str:
    return os.path.abspath(_FILE)


def read_collections() -> List[Dict]:
    path = _abs_path()
    if not os.path.exists(path):
        return []
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []


def _write(collections: List[Dict]) -> None:
    with _lock:
        with open(_abs_path(), "w", encoding="utf-8") as f:
            json.dump(collections, f, indent=2)


def add_collection(
    cid: str,
    filename: str,
    display_name: str,
    status: str = "processing",
) -> None:
    cols = read_collections()
    cols.append({
        "id": cid,
        "filename": filename,
        "displayName": display_name or filename,
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "chunkCount": None,
    })
    _write(cols)


def update_collection_status(
    cid: str,
    status: str,
    chunk_count: Optional[int] = None,
) -> None:
    cols = read_collections()
    for c in cols:
        if c["id"] == cid:
            c["status"] = status
            if chunk_count is not None:
                c["chunkCount"] = chunk_count
            break
    _write(cols)


def remove_collection(cid: str) -> None:
    cols = [c for c in read_collections() if c["id"] != cid]
    _write(cols)


def get_collection(cid: str) -> Optional[Dict]:
    return next((c for c in read_collections() if c["id"] == cid), None)