"""Storage backends."""

from __future__ import annotations

from ..config import StorageSettings
from ..errors import ConfigError
from .base import DenseMatch, IndexStats, LexicalMatch, StorageBackend

__all__ = ["DenseMatch", "IndexStats", "LexicalMatch", "StorageBackend", "open_backend"]


def open_backend(settings: StorageSettings) -> StorageBackend:
    """Construct and initialize the configured backend."""
    if settings.backend == "sqlite":
        from .sqlite.backend import SQLiteBackend

        backend: StorageBackend = SQLiteBackend(settings)
    elif settings.backend == "postgres":
        raise ConfigError(
            "the postgres backend is not implemented yet; use storage.backend: sqlite"
        )
    else:  # pragma: no cover - guarded by the Literal type
        raise ConfigError(f"unknown storage backend: {settings.backend!r}")
    backend.initialize()
    return backend
