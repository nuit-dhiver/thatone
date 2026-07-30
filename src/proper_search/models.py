"""Core data types.

Two families live here:

* **Pipeline types** (:class:`FrameSample`) are plain dataclasses. They carry
  binary payloads, never cross a serialization boundary, and are short-lived.
* **Contract types** (everything else) are Pydantic models. They are validated
  at provider boundaries, persisted, and returned from the public API, so the
  validation cost is worth paying.

:class:`Description` is load-bearing beyond ordinary typing: its JSON schema is
sent to the vision model as the structured-output contract, so changing a field
here changes what the model is asked to produce. Bump ``PROMPT_VERSION`` in
``vision.prompts`` when that happens, or stored rows become uncomparable with
new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    Everything stored is UTC. ``datetime.utcnow()`` is deprecated and returns a
    naive value that silently compares wrong against aware ones.
    """
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class MediaStatus(StrEnum):
    """Where an item sits in the pipeline.

    The order is meaningful: each stage advances the status, so a crashed run
    resumes by selecting rows below the target status.
    """

    PENDING = "pending"
    FETCHED = "fetched"
    SAMPLED = "sampled"
    DESCRIBED = "described"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChunkKind(StrEnum):
    """What part of a description a chunk came from.

    Retrieval weights these differently: a verbatim caption match is a much
    stronger signal than a match against general narrative prose.
    """

    NARRATIVE = "narrative"
    FRAME = "frame"
    SCREEN_TEXT = "screen_text"


class SourceType(StrEnum):
    LOCAL = "local"
    URL = "url"


class JobKind(StrEnum):
    """Pipeline stages, each independently retryable.

    Splitting them means an embedding-provider outage retries only the embed
    step instead of re-paying for vision description.
    """

    FETCH = "fetch"
    DESCRIBE = "describe"
    EMBED = "embed"


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --------------------------------------------------------------------------
# Pipeline types (not persisted)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FrameSample:
    """One frame chosen for description.

    ``image_bytes`` is already encoded (JPEG or PNG) and ready to base64 into a
    request body — encoding once here avoids re-encoding per provider retry.
    """

    index: int
    """Index into the source's decoded frame sequence, not into the sample set."""

    t_ms: int
    """Presentation timestamp in milliseconds from the start of the media."""

    image_bytes: bytes
    media_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    phash: int = 0
    """64-bit dHash. Used for sampling decisions, kept for debugging."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FrameSample(index={self.index}, t_ms={self.t_ms}, "
            f"{self.width}x{self.height}, {len(self.image_bytes)}B)"
        )


@dataclass(slots=True)
class MediaProbe:
    """Container-level facts read during decode, before any sampling."""

    mime: str
    width: int
    height: int
    duration_ms: int
    frame_count: int
    fps: float
    is_animated: bool = True


@dataclass(slots=True)
class MediaRef:
    """A candidate for indexing, before its bytes have been resolved."""

    source_type: SourceType
    source_uri: str
    local_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# The vision output contract
# --------------------------------------------------------------------------


class FrameNote(BaseModel):
    """One frame's observation, timestamped.

    These become individually embedded chunks, which is what lets a query about
    a single moment ("the part where he drops it") match that moment instead of
    competing against a whole-clip summary.
    """

    model_config = ConfigDict(extra="forbid")

    t_ms: int = Field(description="Milliseconds from the start of the clip.")
    note: str = Field(description="What is visible and happening in this frame.")


class Description(BaseModel):
    """What the vision model produces for one item.

    This model's JSON schema is the structured-output contract sent to the
    provider, so field names and descriptions are prompt surface, not just
    documentation.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(
        description=(
            "A detailed account of what happens across the whole clip, in order, "
            "revised to stay consistent with every frame you were shown."
        )
    )
    on_screen_text: str = Field(
        default="",
        description=(
            "Every piece of text visible in the frames, transcribed verbatim, "
            "including captions, subtitles, watermarks, and signs. "
            "Empty string if there is no visible text."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Short lowercase keywords: subjects, actions, setting, emotion, "
            "visual style, notable objects."
        ),
    )
    frame_notes: list[FrameNote] = Field(
        default_factory=list,
        description="One observation per frame you were shown, in time order.",
    )
    confidence: Confidence = Field(
        default=Confidence.MEDIUM,
        description="How confident you are that the narrative is correct.",
    )

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, tags: list[str]) -> list[str]:
        """Lowercase, strip, drop empties, de-duplicate, preserve order.

        Normalizing at the contract boundary rather than at query time means
        the tag filter is an exact index lookup instead of a scan.
        """
        seen: set[str] = set()
        out: list[str] = []
        for tag in tags:
            cleaned = " ".join(tag.lower().split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out

    @field_validator("on_screen_text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    def searchable_text(self) -> str:
        """Flattened representation for the lexical (BM25) index."""
        parts = [self.narrative]
        if self.on_screen_text:
            parts.append(self.on_screen_text)
        if self.tags:
            parts.append(" ".join(self.tags))
        return "\n".join(parts)


# --------------------------------------------------------------------------
# Persisted types
# --------------------------------------------------------------------------


class MediaSource(BaseModel):
    """Where one copy of an item came from.

    An item can have several: the same GIF found on disk and at a URL is one
    ``media`` row with two sources, because identity is the content hash.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_uri: str
    first_seen_at: datetime = Field(default_factory=utcnow)


class MediaItem(BaseModel):
    """One indexed piece of media."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="sha256 of the raw bytes, hex. This is the identity.")
    mime: str
    width: int
    height: int
    duration_ms: int
    frame_count: int
    fps: float
    size_bytes: int
    phash: int = Field(default=0, description="64-bit dHash of the poster frame.")
    thumbnail_path: str | None = None
    status: MediaStatus = MediaStatus.PENDING
    error: str | None = None

    # Provenance — which model and prompt produced the stored description.
    # Needed to decide what a re-index has to redo.
    vision_model: str | None = None
    vision_strategy: str | None = None
    prompt_version: str | None = None

    created_at: datetime = Field(default_factory=utcnow)
    indexed_at: datetime | None = None

    sources: list[MediaSource] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


class Chunk(BaseModel):
    """A unit of text that gets its own embedding."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    media_id: str
    ord: int
    kind: ChunkKind
    text: str
    t_start_ms: int | None = None
    t_end_ms: int | None = None


class SearchHit(BaseModel):
    """One result, with enough detail to explain why it ranked where it did."""

    model_config = ConfigDict(extra="forbid")

    media: MediaItem
    score: float = Field(description="Fused score. Comparable within a result set only.")
    signals: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal ranks/scores (bm25, dense, rerank) for debugging relevance.",
    )
    snippet: str = ""
    snippet_t_ms: int | None = Field(
        default=None,
        description="Timestamp of the best-matching moment, when the match was a frame chunk.",
    )
    matched_chunk_kind: ChunkKind | None = None


class SearchFilters(BaseModel):
    """Hard predicates applied before ranking."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(default_factory=list)
    has_on_screen_text: bool | None = None
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    source_type: SourceType | None = None
    mime: str | None = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.tags,
                self.has_on_screen_text is not None,
                self.min_duration_ms is not None,
                self.max_duration_ms is not None,
                self.source_type is not None,
                self.mime is not None,
            )
        )


class Job(BaseModel):
    """One unit of resumable work."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    media_id: str
    kind: JobKind
    state: JobState = JobState.PENDING
    attempts: int = 0
    last_error: str | None = None
    lease_until: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class UsageRecord(BaseModel):
    """Token accounting for one provider call, for cost reporting."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""

    def __add__(self, other: UsageRecord) -> UsageRecord:
        return UsageRecord(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            model=self.model or other.model,
        )


class DescribeResult(BaseModel):
    """A description plus what it cost to produce."""

    model_config = ConfigDict(extra="forbid")

    description: Description
    usage: UsageRecord = Field(default_factory=UsageRecord)
    model: str = ""
    strategy: str = ""


RerankStrategy = Literal["none", "llm"]
