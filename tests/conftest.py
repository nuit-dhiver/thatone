"""Shared fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from thatone.config import Settings, StorageSettings
from thatone.models import (
    Confidence,
    Description,
    FrameNote,
    MediaItem,
    MediaSource,
    MediaStatus,
    SourceType,
)
from thatone.store.sqlite.backend import SQLiteBackend


@pytest.fixture
def store(tmp_path: Path) -> SQLiteBackend:
    """A fresh, initialized SQLite backend on disk.

    On disk rather than :memory: because thread-local connections each open
    their own handle — an in-memory database would give every thread a
    different, empty database.
    """
    backend = SQLiteBackend(StorageSettings(path=tmp_path / "test.db"))
    backend.initialize()
    yield backend
    backend.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        storage=StorageSettings(path=tmp_path / "test.db", blob_dir=tmp_path / "blobs"),
        vision={"provider": "stub", "model": "stub-1"},
        embedding={"provider": "stub", "model": "stub-embed", "dimensions": 8},
    )


def make_media(
    seed: str = "a",
    *,
    duration_ms: int = 2000,
    mime: str = "image/gif",
    phash: int = 0,
    status: MediaStatus = MediaStatus.PENDING,
) -> MediaItem:
    """A MediaItem with a deterministic content-hash id derived from ``seed``."""
    return MediaItem(
        id=hashlib.sha256(seed.encode()).hexdigest(),
        mime=mime,
        width=480,
        height=270,
        duration_ms=duration_ms,
        frame_count=24,
        fps=12.0,
        size_bytes=100_000,
        phash=phash,
        status=status,
        sources=[MediaSource(source_type=SourceType.LOCAL, source_uri=f"/tmp/{seed}.gif")],
    )


def make_description(
    narrative: str = "A man in a grey suit stands up from a desk and walks away.",
    *,
    on_screen_text: str = "",
    tags: list[str] | None = None,
) -> Description:
    return Description(
        narrative=narrative,
        on_screen_text=on_screen_text,
        tags=tags if tags is not None else ["man", "desk", "office", "walking away"],
        frame_notes=[
            FrameNote(t_ms=0, note="A man sits at a desk looking at the camera."),
            FrameNote(t_ms=900, note="He begins to stand, expression shifting to alarm."),
            FrameNote(t_ms=1800, note="He walks out of frame to the left."),
        ],
        confidence=Confidence.HIGH,
    )
