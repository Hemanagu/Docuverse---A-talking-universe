"""
Stage 2: Dense Search
Re-exports the sync helper and delegates to multi_search for pipeline use.
"""
from rag.multi_search import search_documents, multi_collection_search  # noqa: F401