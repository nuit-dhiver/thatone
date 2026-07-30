"""HTTP request and response models.

Separate from the domain models in :mod:`thatone.models` on purpose. The
wire format is a compatibility surface that outlives any single internal
refactor, and a couple of internal fields — filesystem paths, raw provider
responses — should not leak to clients at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import ChunkKind, MediaItem, MediaStatus, SearchHit, SourceType


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[str] = Field(
        min_length=1,
        description="File paths, directories, or http(s) URLs to index.",
    )
    recursive: bool = True
    wait: bool = Field(
        default=False,
        description=(
            "Run the pipeline before responding. Convenient for a handful of "
            "items; for a large backfill leave this false and let a worker "
            "drain the queue."
        ),
    )


class IndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: int
    media_ids: list[str]
    already_indexed: int = 0
    drained: bool = False


class SourceOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_uri: str
    first_seen_at: datetime


class MediaOut(BaseModel):
    """A media item as returned over HTTP.

    ``thumbnail_path`` is deliberately absent — it is a server-side filesystem
    path. Clients get ``thumbnail_url`` instead.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    mime: str
    width: int
    height: int
    duration_ms: int
    frame_count: int
    fps: float
    size_bytes: int
    status: MediaStatus
    error: str | None = None
    vision_model: str | None = None
    prompt_version: str | None = None
    created_at: datetime
    indexed_at: datetime | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    thumbnail_url: str | None = None
    has_thumbnail: bool = False

    @classmethod
    def from_item(cls, item: MediaItem) -> MediaOut:
        return cls(
            id=item.id,
            mime=item.mime,
            width=item.width,
            height=item.height,
            duration_ms=item.duration_ms,
            frame_count=item.frame_count,
            fps=item.fps,
            size_bytes=item.size_bytes,
            status=item.status,
            error=item.error,
            vision_model=item.vision_model,
            prompt_version=item.prompt_version,
            created_at=item.created_at,
            indexed_at=item.indexed_at,
            sources=[
                SourceOut(
                    source_type=s.source_type,
                    source_uri=s.source_uri,
                    first_seen_at=s.first_seen_at,
                )
                for s in item.sources
            ],
            thumbnail_url=f"/media/{item.id}/thumbnail" if item.thumbnail_path else None,
            has_thumbnail=bool(item.thumbnail_path),
        )


class DescriptionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative: str
    on_screen_text: str = ""
    tags: list[str] = Field(default_factory=list)
    frame_notes: list[dict[str, Any]] = Field(default_factory=list)
    confidence: str = "medium"


class MediaDetailOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media: MediaOut
    description: DescriptionOut | None = None


class HitOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media: MediaOut
    score: float
    snippet: str = ""
    snippet_t_ms: int | None = Field(
        default=None, description="When the matching moment occurs, in milliseconds."
    )
    matched_chunk_kind: ChunkKind | None = None
    signals: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_hit(cls, hit: SearchHit) -> HitOut:
        return cls(
            media=MediaOut.from_item(hit.media),
            score=hit.score,
            snippet=hit.snippet,
            snippet_t_ms=hit.snippet_t_ms,
            matched_chunk_kind=hit.matched_chunk_kind,
            signals=hit.signals,
        )


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    count: int
    hits: list[HitOut]
    candidates_by_signal: dict[str, int] = Field(default_factory=dict)
    degraded_signals: dict[str, str] | None = Field(
        default=None,
        description=(
            "Signals that failed and were skipped. Present means the results "
            "are real but incomplete — surface it rather than silently "
            "returning a worse result set."
        ),
    )


class JobOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None
    media_id: str
    kind: str
    state: str
    attempts: int
    last_error: str | None = None
    updated_at: datetime


class StatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_total: int
    media_by_status: dict[str, int] = Field(default_factory=dict)
    chunk_total: int
    vector_total: int
    tag_total: int
    jobs_by_state: dict[str, int] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dim: int | None = None


class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[str] = Field(min_length=1)
    sample_size: int = Field(default=5, ge=1, le=50)


class EstimateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_count: int
    sampled: int
    model: str
    strategy: str
    batch: bool
    measured: bool = Field(
        description="True when token counts came from the provider rather than an estimate."
    )
    avg_frames: float
    avg_input_tokens: float
    cost_per_item: float
    vision_cost: float
    embedding_cost: float
    total_cost: float
    warnings: list[str] = Field(default_factory=list)
    summary: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    storage: str
    vision_provider: str | None = None
    embedding_provider: str | None = None
    indexing_available: bool
    dense_search_available: bool


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    kind: str
