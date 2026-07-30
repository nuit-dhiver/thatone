"""The indexing pipeline, one stage at a time.

Three stages, deliberately separate, because they fail for different reasons
and cost different amounts:

``ingest``
    Hash, probe, sample-scan, thumbnail, record. Cheap, local, no network.
``describe``
    The vision call. By far the most expensive step, and the one that must
    never be repeated because a later stage failed.
``embed``
    Chunk, embed, store vectors. Cheap per item but network-bound.

Keeping them apart is what makes a 100k-item run survivable: an embedding
provider outage retries only ``embed``, and the vision spend already banked
stays banked.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .embed.base import EmbeddingProvider, InputType
from .embed.chunking import build_chunks
from .errors import MediaError, MediaTooLargeError, TerminalError, ThatOneError
from .media.decode import decode_at_indices, probe
from .media.hashing import content_hash_file
from .media.sampling import FrameMeta, sample_frames, scan_timeline
from .media.thumbnail import write_poster, write_preview
from .models import (
    JobKind,
    MediaItem,
    MediaSource,
    MediaStatus,
    SourceType,
    UsageRecord,
)
from .store.base import StorageBackend
from .vision.base import VisionProvider
from .vision.prompts import PROMPT_VERSION
from .vision.strategies import get_strategy

log = logging.getLogger(__name__)


class Indexer:
    """Drives one item through the pipeline."""

    def __init__(
        self,
        settings: Settings,
        store: StorageBackend,
        vision: VisionProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        *,
        caption_vision: VisionProvider | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.vision = vision
        self.embedder = embedder
        self.caption_vision = caption_vision
        self.blob_dir = Path(settings.storage.blob_dir).expanduser()

    # -- stage 1: ingest ---------------------------------------------------

    async def ingest_path(
        self, path: str | Path, *, source_type: SourceType = SourceType.LOCAL
    ) -> tuple[MediaItem, bool]:
        """Register a local file. Returns ``(item, is_new)``.

        Identity is the content hash, so re-ingesting a file already indexed
        under a different name adds a source and returns ``is_new=False``
        rather than paying for a second description.
        """
        return await asyncio.to_thread(self._ingest_path_sync, Path(path), source_type)

    def _ingest_path_sync(
        self, path: Path, source_type: SourceType
    ) -> tuple[MediaItem, bool]:
        if not path.is_file():
            raise MediaError(f"not a file: {path}")

        size = path.stat().st_size
        if size > self.settings.fetch.max_bytes:
            raise MediaTooLargeError(
                f"{path} is {size} bytes, over the fetch.max_bytes limit of "
                f"{self.settings.fetch.max_bytes}"
            )

        media_id = content_hash_file(path)
        source = MediaSource(source_type=source_type, source_uri=str(path.resolve()))

        existing = self.store.get_media(media_id)
        if existing is not None:
            self.store.add_source(media_id, source)
            return existing, False

        info = probe(path)
        if info.duration_ms > self.settings.fetch.max_duration_ms:
            raise MediaTooLargeError(
                f"{path} runs {info.duration_ms}ms, over the fetch.max_duration_ms "
                f"limit of {self.settings.fetch.max_duration_ms}"
            )

        # The timeline is the authority on frame count and duration; container
        # metadata is frequently wrong for GIFs.
        timeline = scan_timeline(path, max_decode_frames=self.settings.sampling.max_decode_frames)
        duration_ms = max(info.duration_ms, timeline[-1].t_ms if timeline else 0)

        thumbnail_path = self._write_thumbnails(path, media_id, timeline)

        item = MediaItem(
            id=media_id,
            mime=info.mime,
            width=info.width,
            height=info.height,
            duration_ms=duration_ms,
            frame_count=len(timeline),
            fps=info.fps,
            size_bytes=size,
            phash=timeline[0].phash if timeline else 0,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            status=MediaStatus.SAMPLED,
            sources=[source],
        )
        created = self.store.upsert_media(item)
        self.store.enqueue(media_id, JobKind.DESCRIBE)
        return item, created

    def _write_thumbnails(
        self, path: Path, media_id: str, timeline: Sequence[FrameMeta]
    ) -> Path | None:
        """Poster plus animated preview. Never fatal — a missing thumbnail is a
        cosmetic problem, and failing ingest over it would be worse."""
        try:
            indices = [m.index for m in timeline[:24]]
            frames = decode_at_indices(path, indices)
            if not frames:
                return None
            poster = write_poster(frames[0].image, self.blob_dir, media_id)
            write_preview(frames, self.blob_dir, media_id)
            return poster
        except Exception as exc:  # pragma: no cover - cosmetic path
            log.warning("thumbnail generation failed for %s: %s", media_id, exc)
            return None

    # -- stage 2: describe -------------------------------------------------

    async def describe(self, media_id: str) -> UsageRecord:
        """Run the vision pass and store the result."""
        if self.vision is None:
            raise ThatOneError("no vision provider configured")

        item = self.store.get_media(media_id)
        if item is None:
            raise MediaError(f"unknown media: {media_id}")

        path = await asyncio.to_thread(self._resolve_path, item)
        frames = await asyncio.to_thread(sample_frames, path, self.settings.sampling)

        result = await get_strategy(self.settings.vision.strategy).run(
            self.vision,
            frames,
            duration_ms=item.duration_ms,
            caption_provider=self.caption_vision,
        )

        await asyncio.to_thread(
            self.store.save_description,
            media_id,
            result.description,
            model=result.model,
            strategy=result.strategy,
            prompt_version=PROMPT_VERSION,
        )
        await asyncio.to_thread(self.store.enqueue, media_id, JobKind.EMBED)
        return result.usage

    # -- stage 3: embed ----------------------------------------------------

    async def embed(self, media_id: str) -> int:
        """Chunk the description, embed it, and mark the item searchable."""
        if self.embedder is None:
            raise ThatOneError("no embedding provider configured")

        description = await asyncio.to_thread(self.store.get_description, media_id)
        if description is None:
            raise MediaError(f"{media_id} has no description to embed")

        await asyncio.to_thread(
            self.store.ensure_vector_space,
            model=self.embedder.model,
            dimensions=self.embedder.dimensions,
        )

        chunks = build_chunks(
            media_id, description, max_chars=self.settings.embedding.max_chunk_chars
        )
        if not chunks:
            # Nothing to embed, but the item is still lexically searchable, so
            # it is indexed rather than failed.
            await asyncio.to_thread(self.store.set_status, media_id, MediaStatus.INDEXED)
            return 0

        chunk_ids = await asyncio.to_thread(self.store.replace_chunks, media_id, chunks)
        vectors = await self.embedder.embed(
            [c.text for c in chunks], input_type=InputType.DOCUMENT
        )
        await asyncio.to_thread(
            self.store.save_embeddings, list(zip(chunk_ids, vectors, strict=True))
        )
        await asyncio.to_thread(self.store.set_status, media_id, MediaStatus.INDEXED)
        return len(chunks)

    # -- helpers -----------------------------------------------------------

    def _resolve_path(self, item: MediaItem) -> Path:
        """Find the bytes for an item on disk.

        Prefers a local source, then the blob cache. A URL-sourced item whose
        cached blob has been evicted is a terminal failure for the description
        stage: the frames cannot be re-derived without re-fetching, which is
        the ingest stage's job, not this one's.
        """
        for source in item.sources:
            if source.source_type is SourceType.LOCAL:
                candidate = Path(source.source_uri)
                if candidate.is_file():
                    return candidate

        cached = self.blob_dir / item.id[:2] / item.id
        if cached.is_file():
            return cached

        raise MediaError(
            f"no readable copy of {item.id} remains: local sources are missing and "
            f"there is no cached blob at {cached}. Re-ingest the source to restore it."
        )

    async def run_all(self, media_id: str) -> None:
        """Describe then embed. Convenience for tests and single-item flows."""
        await self.describe(media_id)
        await self.embed(media_id)

    async def index_paths(
        self, paths: Sequence[str | Path], *, concurrency: int | None = None
    ) -> list[MediaItem]:
        """Ingest, describe, and embed a batch of local files.

        Bounded concurrency, because the ceiling here is the provider's rate
        limit rather than local CPU. A per-item failure is recorded against
        that item and does not stop the batch — the whole point of a resumable
        pipeline is that one bad file does not end a 100k-item run.
        """
        limit = concurrency or self.settings.jobs.concurrency
        semaphore = asyncio.Semaphore(limit)
        results: list[MediaItem] = []

        async def one(path: str | Path) -> None:
            async with semaphore:
                try:
                    item, _ = await self.ingest_path(path)
                    await self.run_all(item.id)
                    refreshed = await asyncio.to_thread(self.store.get_media, item.id)
                    if refreshed is not None:
                        results.append(refreshed)
                except TerminalError as exc:
                    log.warning("skipping %s: %s", path, exc)
                except ThatOneError as exc:
                    log.error("failed on %s: %s", path, exc)

        await asyncio.gather(*(one(p) for p in paths))
        return results
