"""proper-search — find GIFs and videos by what happens inside them.

A vision model watches sampled frames, writes down what happens, and that
writing is indexed for hybrid keyword + semantic retrieval — so a half-
remembered fragment ("the one where he slowly turns around", "it said NOPE")
finds the clip.

This package never runs a model. Every provider is an HTTP client; to use a
self-hosted model, run it on your own server and point ``vision.base_url`` at
it.
"""

from __future__ import annotations

from .config import Settings
from .engine import ProperSearch
from .errors import (
    ConfigError,
    DecodeError,
    IndexConsistencyError,
    MediaError,
    ProperSearchError,
    ProviderError,
    ProviderRefusal,
    RetryableError,
    TerminalError,
)
from .models import (
    Chunk,
    ChunkKind,
    Description,
    FrameNote,
    FrameSample,
    JobKind,
    JobState,
    MediaItem,
    MediaStatus,
    SearchFilters,
    SearchHit,
    SourceType,
    UsageRecord,
)

__version__ = "0.1.0"

# Grouped by concern rather than sorted alphabetically: readers scanning the
# public surface care what a name is for, not what letter it starts with.
__all__ = [  # noqa: RUF022
    "__version__",
    # Entry point
    "ProperSearch",
    # Configuration
    "Settings",
    # Models
    "Chunk",
    "ChunkKind",
    "Description",
    "FrameNote",
    "FrameSample",
    "JobKind",
    "JobState",
    "MediaItem",
    "MediaStatus",
    "SearchFilters",
    "SearchHit",
    "SourceType",
    "UsageRecord",
    # Errors
    "ConfigError",
    "DecodeError",
    "IndexConsistencyError",
    "MediaError",
    "ProperSearchError",
    "ProviderError",
    "ProviderRefusal",
    "RetryableError",
    "TerminalError",
]
