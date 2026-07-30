"""Thumbnail generation.

Two artefacts per item: a still poster frame, and an animated WebP preview.
Both are written under a content-hash-sharded directory so a single flat folder
never accumulates 100k entries — most filesystems cope, but directory listings
and backups stop being usable well before that.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from PIL import Image

from .decode import DecodedFrame

POSTER_MAX_EDGE = 480
PREVIEW_MAX_EDGE = 320
PREVIEW_MAX_FRAMES = 24


def thumbnail_dir(root: str | Path, media_id: str) -> Path:
    """Shard by the first two hex characters of the content hash."""
    return Path(root) / media_id[:2]


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def write_poster(
    image: Image.Image, root: str | Path, media_id: str, *, max_edge: int = POSTER_MAX_EDGE
) -> Path:
    """Write a still poster frame as WebP. Returns the path."""
    directory = thumbnail_dir(root, media_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{media_id}.webp"
    _fit(image.convert("RGB"), max_edge).save(path, format="WEBP", quality=80, method=4)
    return path


def write_preview(
    frames: Sequence[DecodedFrame],
    root: str | Path,
    media_id: str,
    *,
    max_edge: int = PREVIEW_MAX_EDGE,
    max_frames: int = PREVIEW_MAX_FRAMES,
) -> Path | None:
    """Write an animated WebP preview, or None if there is nothing to animate.

    Frame durations are taken from the decoded timestamps rather than assumed
    uniform, so a GIF with variable delays previews at its real pacing.
    """
    if len(frames) < 2:
        return None

    ordered = sorted(frames, key=lambda f: f.index)
    if len(ordered) > max_frames:
        step = (len(ordered) - 1) / (max_frames - 1)
        ordered = [ordered[round(i * step)] for i in range(max_frames)]

    images = [_fit(f.image.convert("RGB"), max_edge) for f in ordered]

    durations: list[int] = []
    for current, following in pairwise(ordered):
        durations.append(max(20, following.t_ms - current.t_ms))
    durations.append(durations[-1] if durations else 100)

    directory = thumbnail_dir(root, media_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{media_id}.preview.webp"
    images[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        quality=70,
        method=4,
    )
    return path


def encode_poster_bytes(image: Image.Image, *, max_edge: int = POSTER_MAX_EDGE) -> bytes:
    """Poster frame as WebP bytes, for callers serving it without touching disk."""
    buffer = io.BytesIO()
    _fit(image.convert("RGB"), max_edge).save(buffer, format="WEBP", quality=80, method=4)
    return buffer.getvalue()
