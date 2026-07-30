"""SQLite backend: FTS5 for lexical retrieval, sqlite-vec for dense."""

from .backend import SQLiteBackend, build_fts_query

__all__ = ["SQLiteBackend", "build_fts_query"]
