"""Storage backend interface.

Every persistence concern goes through this ABC so the SQLite default and the
Postgres adapter stay interchangeable. The interface is deliberately concrete
about *retrieval* — :meth:`search_lexical` and :meth:`search_dense` return
ranked candidate lists rather than exposing a query builder — because fusion
happens above this layer and needs comparable shapes from both signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import (
    Chunk,
    Description,
    Job,
    JobKind,
    MediaItem,
    MediaSource,
    MediaStatus,
    SearchFilters,
)


@dataclass(slots=True, frozen=True)
class LexicalMatch:
    """One BM25 hit.

    ``score`` is already sign-flipped so larger is better: FTS5's ``bm25()``
    returns negative values where more negative means more relevant, which
    inverts against every other signal and is a reliable source of ranking bugs.
    """

    media_id: str
    score: float
    chunk_id: int | None = None
    snippet: str = ""


@dataclass(slots=True, frozen=True)
class DenseMatch:
    """One vector hit, resolved to the chunk that matched.

    A media item can appear several times, once per matching chunk; collapsing
    to the best chunk per item is the caller's job, since that is a ranking
    decision rather than a storage one.
    """

    media_id: str
    chunk_id: int
    distance: float
    """Lower is better (L2 by default)."""


@dataclass(slots=True, frozen=True)
class IndexStats:
    media_total: int = 0
    media_by_status: dict[str, int] | None = None
    chunk_total: int = 0
    vector_total: int = 0
    tag_total: int = 0
    jobs_by_state: dict[str, int] | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None


class StorageBackend(ABC):
    """Persistence and retrieval primitives."""

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Create the schema if absent and run pending migrations.

        Must be idempotent: called on every process start, not just first run.
        """

    @abstractmethod
    def close(self) -> None: ...

    # -- media -------------------------------------------------------------

    @abstractmethod
    def upsert_media(self, item: MediaItem) -> bool:
        """Insert or update by content hash. Returns True if newly created.

        The return value drives dedupe: the same GIF arriving from a second
        source updates sources without re-paying for description.
        """

    @abstractmethod
    def get_media(self, media_id: str) -> MediaItem | None: ...

    @abstractmethod
    def get_media_many(self, media_ids: Sequence[str]) -> dict[str, MediaItem]:
        """Batch fetch. Search resolves a whole result page in one round trip."""

    @abstractmethod
    def find_by_source_uri(self, source_uri: str) -> str | None:
        """Resolve a source URI to a media id, so a re-scan can skip the fetch."""

    @abstractmethod
    def add_source(self, media_id: str, source: MediaSource) -> None: ...

    @abstractmethod
    def set_status(
        self, media_id: str, status: MediaStatus, *, error: str | None = None
    ) -> None: ...

    @abstractmethod
    def list_media(
        self,
        *,
        status: MediaStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MediaItem]: ...

    @abstractmethod
    def find_near_duplicates(self, phash: int, *, max_distance: int = 4) -> list[str]:
        """Media whose perceptual hash is within ``max_distance`` bits.

        Distinct from content-hash identity: this catches the same GIF
        re-encoded at a different quality or resolution, which hashes
        differently but should not be described twice.
        """

    @abstractmethod
    def delete_media(self, media_id: str) -> bool: ...

    # -- descriptions ------------------------------------------------------

    @abstractmethod
    def save_description(
        self,
        media_id: str,
        description: Description,
        *,
        model: str,
        strategy: str,
        prompt_version: str,
        raw_response: str | None = None,
    ) -> None:
        """Persist a description and refresh every index derived from it.

        Must be atomic across the description row, the tag rows, and the
        lexical index — a partial write leaves an item searchable under stale
        text, which is worse than not being searchable at all.
        """

    @abstractmethod
    def get_description(self, media_id: str) -> Description | None: ...

    # -- chunks and vectors ------------------------------------------------

    @abstractmethod
    def replace_chunks(self, media_id: str, chunks: Sequence[Chunk]) -> list[int]:
        """Replace all chunks for an item. Returns the assigned chunk ids.

        Replace rather than append so re-describing cannot leave orphaned
        chunks pointing at superseded text.
        """

    @abstractmethod
    def get_chunks(self, media_id: str) -> list[Chunk]: ...

    @abstractmethod
    def get_chunk(self, chunk_id: int) -> Chunk | None: ...

    @abstractmethod
    def ensure_vector_space(self, *, model: str, dimensions: int) -> None:
        """Bind the index to one embedding model and width.

        First call records the pair; later calls verify it. A mismatch raises
        :class:`~proper_search.errors.IndexConsistencyError` rather than
        proceeding, because comparing vectors from different models yields
        confident nonsense instead of an obvious failure.
        """

    @abstractmethod
    def save_embeddings(self, pairs: Sequence[tuple[int, Sequence[float]]]) -> None:
        """Store ``(chunk_id, vector)`` pairs."""

    @abstractmethod
    def chunks_missing_embeddings(self, *, limit: int = 500) -> list[Chunk]: ...

    # -- retrieval ---------------------------------------------------------

    @abstractmethod
    def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
        weights: tuple[float, float, float] = (1.0, 3.0, 1.5),
    ) -> list[LexicalMatch]:
        """BM25 over narrative, on-screen text, and tags."""

    @abstractmethod
    def search_chunks_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[LexicalMatch]:
        """BM25 over chunk text, for moment-level keyword matches."""

    @abstractmethod
    def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[DenseMatch]:
        """K-nearest-neighbour over chunk embeddings."""

    @abstractmethod
    def filter_media_ids(self, filters: SearchFilters, *, limit: int = 10_000) -> set[str] | None:
        """Media ids satisfying ``filters``, or None when no filter applies."""

    # -- jobs --------------------------------------------------------------

    @abstractmethod
    def enqueue(self, media_id: str, kind: JobKind) -> int:
        """Add a job, or return the existing id if one is already queued."""

    @abstractmethod
    def claim_jobs(self, kind: JobKind, *, limit: int, lease_seconds: int) -> list[Job]:
        """Atomically claim up to ``limit`` runnable jobs.

        Claiming sets a lease. An expired lease is reclaimable, which is what
        lets a killed worker's in-flight work resume instead of stranding.
        """

    @abstractmethod
    def complete_job(self, job_id: int) -> None: ...

    @abstractmethod
    def fail_job(
        self, job_id: int, error: str, *, retry: bool, backoff_seconds: float = 0.0
    ) -> None:
        """Record a failure. ``retry=False`` marks it terminal."""

    @abstractmethod
    def reclaim_expired_leases(self) -> int:
        """Return expired claims to the pending pool. Returns the count."""

    @abstractmethod
    def list_jobs(
        self, *, state: str | None = None, kind: JobKind | None = None, limit: int = 100
    ) -> list[Job]: ...

    # -- introspection -----------------------------------------------------

    @abstractmethod
    def stats(self) -> IndexStats: ...

    @abstractmethod
    def get_meta(self, key: str) -> str | None: ...

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None: ...

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> StorageBackend:
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
