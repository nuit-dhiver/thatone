"""The public library surface.

:class:`ThatOne` owns the wiring — storage, providers, indexer, worker,
search pipeline — so callers assemble nothing by hand. The HTTP layer is a thin
wrapper over exactly these methods rather than a parallel implementation, which
is what keeps the two from drifting apart as either one changes.

Resource ownership is explicit. Providers hold HTTP clients and the store holds
database handles, so this is a context manager and :meth:`close` is not
optional::

    async with ThatOne.open(Settings()) as engine:
        await engine.index(["clips/"])
        hits, _ = await engine.search("cat knocking a glass off a table")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .embed import build_embedding_provider
from .embed.base import EmbeddingProvider
from .errors import ConfigError, TerminalError, ThatOneError
from .estimate import CostEstimate, CostEstimator
from .indexer import Indexer
from .ingest.fetch import MediaFetcher
from .ingest.sources import iter_local_media
from .jobs.worker import Worker, WorkerStats
from .models import (
    JobKind,
    MediaItem,
    MediaSource,
    MediaStatus,
    SearchFilters,
    SearchHit,
    SourceType,
)
from .search.pipeline import SearchDiagnostics, SearchPipeline
from .store import open_backend
from .store.base import IndexStats, StorageBackend
from .vision import build_caption_provider, build_provider
from .vision.base import VisionProvider

log = logging.getLogger(__name__)


class ThatOne:
    """A configured index: ingest, describe, embed, search."""

    def __init__(
        self,
        settings: Settings,
        store: StorageBackend,
        vision: VisionProvider | None,
        embedder: EmbeddingProvider | None,
        *,
        caption_vision: VisionProvider | None = None,
        reranker: VisionProvider | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.vision = vision
        self.embedder = embedder
        self.caption_vision = caption_vision
        self.indexer = Indexer(settings, store, vision, embedder, caption_vision=caption_vision)
        self.worker = Worker(settings, store, self.indexer)
        self.searcher = SearchPipeline(store, embedder, settings.search, reranker=reranker)
        self.estimator = CostEstimator(settings, vision)
        self._closed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, settings: Settings | None = None) -> ThatOne:
        """Build every component from configuration.

        Provider construction is allowed to fail softly: a missing embedding
        key should still leave a usable read-only index for lexical search
        rather than refusing to open at all.
        """
        settings = settings or Settings()
        store = open_backend(settings.storage)

        vision: VisionProvider | None = None
        caption_vision: VisionProvider | None = None
        embedder: EmbeddingProvider | None = None

        try:
            vision = build_provider(settings)
            caption_vision = build_caption_provider(settings)
        except (ConfigError, ImportError) as exc:
            log.warning("vision provider unavailable, indexing disabled: %s", exc)

        try:
            embedder = build_embedding_provider(settings)
        except (ConfigError, ImportError) as exc:
            log.warning("embedding provider unavailable, dense search disabled: %s", exc)

        reranker = vision if settings.search.rerank == "llm" else None
        return cls(
            settings, store, vision, embedder,
            caption_vision=caption_vision, reranker=reranker,
        )

    # -- indexing ----------------------------------------------------------

    async def index(
        self,
        targets: Sequence[str | Path],
        *,
        recursive: bool = True,
        drain: bool = True,
    ) -> list[MediaItem]:
        """Ingest paths, directories, or URLs, then run the pipeline.

        ``drain=False`` registers the work and returns immediately, leaving the
        queue for a worker to pick up — the right shape for a large backfill,
        where waiting on 100k items in one call is not a real option.
        """
        refs = await asyncio.to_thread(self._expand, list(targets), recursive)
        items: list[MediaItem] = []

        for source_type, target in refs:
            try:
                if source_type is SourceType.URL:
                    item, _ = await self.ingest_url(str(target))
                else:
                    item, _ = await self.indexer.ingest_path(target)
                items.append(item)
            except TerminalError as exc:
                log.warning("skipping %s: %s", target, exc)
            except ThatOneError as exc:
                log.error("failed to ingest %s: %s", target, exc)

        if drain:
            await self.worker.drain()
            refreshed = await asyncio.to_thread(
                self.store.get_media_many, [i.id for i in items]
            )
            return [refreshed[i.id] for i in items if i.id in refreshed]
        return items

    def _expand(
        self, targets: Sequence[str | Path], recursive: bool
    ) -> list[tuple[SourceType, str | Path]]:
        """Resolve directories to their contents and classify each target."""
        out: list[tuple[SourceType, str | Path]] = []
        for target in targets:
            text = str(target)
            if text.startswith(("http://", "https://")):
                out.append((SourceType.URL, text))
                continue
            path = Path(text).expanduser()
            if path.is_dir():
                out.extend(
                    (SourceType.LOCAL, ref.source_uri)
                    for ref in iter_local_media(path, recursive=recursive)
                )
            else:
                out.append((SourceType.LOCAL, path))
        return out

    async def ingest_url(self, url: str) -> tuple[MediaItem, bool]:
        """Download a URL into the blob cache and register it.

        Dedupe happens on the downloaded bytes, so the same GIF served from two
        URLs is one item with two sources — described, and paid for, once.
        """
        known = await asyncio.to_thread(self.store.find_by_source_uri, url)
        if known:
            existing = await asyncio.to_thread(self.store.get_media, known)
            if existing is not None:
                return existing, False

        async with MediaFetcher(self.settings.fetch, self.settings.storage.blob_dir) as fetcher:
            result = await fetcher.fetch(url)

        item, created = await self.indexer.ingest_path(result.path, source_type=SourceType.URL)
        await asyncio.to_thread(
            self.store.add_source,
            item.id,
            MediaSource(source_type=SourceType.URL, source_uri=url),
        )
        return item, created

    async def reindex(self, media_id: str, *, redescribe: bool = True) -> MediaItem | None:
        """Re-run the pipeline for one item.

        ``redescribe=False`` re-chunks and re-embeds from the stored
        description, which is the cheap path when only the embedding model or
        chunking changed — no vision spend at all.
        """
        if redescribe:
            await asyncio.to_thread(self.store.enqueue, media_id, JobKind.DESCRIBE)
            await self.indexer.describe(media_id)
        await self.indexer.embed(media_id)
        return await asyncio.to_thread(self.store.get_media, media_id)

    async def drain(self) -> WorkerStats:
        """Run queued work until the queue is empty."""
        return await self.worker.drain()

    # -- search ------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: SearchFilters | None = None,
        explain: bool = False,
    ) -> tuple[list[SearchHit], SearchDiagnostics]:
        """Find clips matching a description of what happens in them."""
        return await self.searcher.search(query, limit=limit, filters=filters, explain=explain)

    # -- introspection -----------------------------------------------------

    async def get(self, media_id: str) -> MediaItem | None:
        return await asyncio.to_thread(self.store.get_media, media_id)

    async def list_media(
        self, *, status: MediaStatus | None = None, limit: int = 50, offset: int = 0
    ) -> list[MediaItem]:
        return await asyncio.to_thread(
            self.store.list_media, status=status, limit=limit, offset=offset
        )

    async def stats(self) -> IndexStats:
        return await asyncio.to_thread(self.store.stats)

    async def estimate(
        self, targets: Sequence[str | Path], *, sample_size: int = 5
    ) -> CostEstimate:
        """Project the cost of describing a corpus before committing to it."""
        refs = await asyncio.to_thread(self._expand, list(targets), True)
        paths = [t for kind, t in refs if kind is SourceType.LOCAL]
        return await self.estimator.estimate_paths(list(paths), sample_size=sample_size)

    # -- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for provider in (self.vision, self.caption_vision, self.embedder):
            if provider is not None:
                try:
                    await provider.close()
                except Exception as exc:  # pragma: no cover - teardown must not mask errors
                    log.warning("error closing %s: %s", provider, exc)
        await asyncio.to_thread(self.store.close)

    async def __aenter__(self) -> ThatOne:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
