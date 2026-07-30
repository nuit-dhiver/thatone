"""FastAPI application.

A thin wrapper over :class:`~proper_search.engine.ProperSearch`. Routes parse
input, call one engine method, and shape the response — no business logic lives
here, so the library and the HTTP surface cannot drift.

The engine is built once in the lifespan handler and shared: it owns a database
handle and pooled HTTP clients, and rebuilding it per request would leak both.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse

from .. import __version__
from ..config import Settings
from ..engine import ProperSearch
from ..errors import (
    AuthError,
    ConfigError,
    IndexConsistencyError,
    MediaTooLargeError,
    ProperSearchError,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderUnavailable,
    RetryableError,
    StorageError,
    TerminalError,
)
from ..models import MediaStatus, SearchFilters, SourceType
from .schemas import (
    DescriptionOut,
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    HitOut,
    IndexRequest,
    IndexResponse,
    JobOut,
    MediaDetailOut,
    MediaOut,
    SearchResponse,
    StatsResponse,
)

log = logging.getLogger(__name__)

# Which failures map to which status. The distinction that matters is 4xx for
# "your request or your media is the problem" versus 5xx for "something upstream
# is the problem" — a client retrying the first will always fail, and a client
# retrying the second may well succeed.
#
# ORDER IS LOAD-BEARING. Several errors inherit from two bases: ProviderRefusal
# is both a TerminalError and a ProviderError, and AuthError is too. Matching
# the generic ProviderError first would map a refusal to 502, telling the client
# to retry something that will be declined identically every time. Concrete
# types are therefore listed before the classification bases they derive from.
#
# Literal codes rather than the starlette constants, several of which are being
# renamed and emit deprecation warnings on import.
_STATUS_MAP: list[tuple[type[Exception], int]] = [
    # Specific conditions, most actionable first.
    (IndexConsistencyError, 409),
    (MediaTooLargeError, 413),
    (AuthError, 502),  # our credentials are wrong; the client can do nothing
    (ProviderRefusal, 422),  # this media, permanently — never retry
    (ProviderRateLimited, 429),
    (ProviderUnavailable, 503),
    (ConfigError, 500),
    (StorageError, 500),
    # Classification fallbacks for anything not named above.
    (RetryableError, 503),
    (TerminalError, 422),
    (ProviderError, 502),
]


def _status_for(exc: Exception) -> int:
    for kind, code in _STATUS_MAP:
        if isinstance(exc, kind):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def get_engine(request: Request) -> ProperSearch:
    engine: ProperSearch | None = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(status_code=503, detail="engine is not initialized")
    return engine


EngineDep = Annotated[ProperSearch, Depends(get_engine)]


def create_app(settings: Settings | None = None, *, engine: ProperSearch | None = None) -> FastAPI:
    """Build the application.

    ``engine`` can be injected so tests drive a fully stubbed pipeline without
    the app reaching for real providers or a real database.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_engine = engine is None
        app.state.engine = engine or ProperSearch.open(settings or Settings())
        try:
            yield
        finally:
            # Only tear down what we created; an injected engine belongs to
            # whoever passed it in.
            if owns_engine:
                await app.state.engine.close()

    app = FastAPI(
        title="proper-search",
        version=__version__,
        summary="Search GIFs and videos by what happens inside them.",
        lifespan=lifespan,
    )

    @app.exception_handler(ProperSearchError)
    async def handle_domain_error(_: Request, exc: ProperSearchError) -> JSONResponse:
        code = _status_for(exc)
        if code >= 500:
            log.exception("request failed", exc_info=exc)
        return JSONResponse(
            status_code=code,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "kind": "terminal" if isinstance(exc, TerminalError) else "transient",
            },
        )

    # -- health ------------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz(engine: EngineDep) -> HealthResponse:
        """Liveness plus which capabilities are actually available.

        Reports degraded capability rather than failing: an index with no
        embedding key still serves lexical search perfectly well, and an
        operator needs to see *which* half is missing.
        """
        return HealthResponse(
            status="ok",
            version=__version__,
            storage=engine.settings.storage.backend,
            vision_provider=engine.vision.name if engine.vision else None,
            embedding_provider=engine.embedder.name if engine.embedder else None,
            indexing_available=engine.vision is not None and engine.embedder is not None,
            dense_search_available=engine.embedder is not None,
        )

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def get_stats(engine: EngineDep) -> StatsResponse:
        stats = await engine.stats()
        return StatsResponse(
            media_total=stats.media_total,
            media_by_status=stats.media_by_status or {},
            chunk_total=stats.chunk_total,
            vector_total=stats.vector_total,
            tag_total=stats.tag_total,
            jobs_by_state=stats.jobs_by_state or {},
            embedding_model=stats.embedding_model,
            embedding_dim=stats.embedding_dim,
        )

    # -- indexing ----------------------------------------------------------

    @app.post(
        "/index",
        response_model=IndexResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["index"],
    )
    async def index(request: IndexRequest, engine: EngineDep) -> IndexResponse:
        """Ingest paths, directories, or URLs.

        202 rather than 201 because the default path only registers the work;
        describing and embedding happen on the queue.
        """
        before = (await engine.stats()).media_total
        items = await engine.index(
            request.targets, recursive=request.recursive, drain=request.wait
        )
        after = (await engine.stats()).media_total
        created = max(0, after - before)
        return IndexResponse(
            accepted=len(items),
            media_ids=[i.id for i in items],
            already_indexed=max(0, len(items) - created),
            drained=request.wait,
        )

    @app.post("/drain", tags=["index"])
    async def drain(engine: EngineDep) -> dict[str, Any]:
        """Run queued work until the queue is empty."""
        stats = await engine.drain()
        return {
            "claimed": stats.claimed,
            "completed": stats.completed,
            "retried": stats.retried,
            "failed": stats.failed,
            "input_tokens": stats.usage.input_tokens,
            "output_tokens": stats.usage.output_tokens,
        }

    @app.post("/estimate", response_model=EstimateResponse, tags=["index"])
    async def estimate(request: EstimateRequest, engine: EngineDep) -> EstimateResponse:
        """Project what describing a corpus would cost, before running it."""
        result = await engine.estimate(request.targets, sample_size=request.sample_size)
        return EstimateResponse(
            item_count=result.item_count,
            sampled=result.sampled,
            model=result.model,
            strategy=result.strategy,
            batch=result.batch,
            measured=result.measured,
            avg_frames=round(result.avg_frames, 2),
            avg_input_tokens=round(result.avg_input_tokens, 1),
            cost_per_item=result.cost_per_item,
            vision_cost=result.vision_cost,
            embedding_cost=result.embedding_cost,
            total_cost=result.total_cost,
            warnings=result.warnings,
            summary=result.summary(),
        )

    # -- search ------------------------------------------------------------

    @app.get("/search", response_model=SearchResponse, tags=["search"])
    async def search(
        engine: EngineDep,
        q: Annotated[str, Query(description="What you remember about the clip.")],
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        tags: Annotated[list[str] | None, Query(description="All must be present.")] = None,
        has_text: Annotated[bool | None, Query(description="Filter on burned-in text.")] = None,
        min_duration_ms: Annotated[int | None, Query(ge=0)] = None,
        max_duration_ms: Annotated[int | None, Query(ge=0)] = None,
        mime: Annotated[str | None, Query()] = None,
        source_type: Annotated[SourceType | None, Query()] = None,
        explain: Annotated[bool, Query(description="Include per-signal ranks.")] = False,
    ) -> SearchResponse:
        """Find clips by what happens inside them."""
        filters = SearchFilters(
            tags=tags or [],
            has_on_screen_text=has_text,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            mime=mime,
            source_type=source_type,
        )
        hits, diagnostics = await engine.search(
            q, limit=limit, filters=None if filters.is_empty() else filters, explain=explain
        )
        return SearchResponse(
            query=q,
            count=len(hits),
            hits=[HitOut.from_hit(h) for h in hits],
            candidates_by_signal=diagnostics.candidates_by_signal,
            degraded_signals=diagnostics.degraded_signals,
        )

    # -- media -------------------------------------------------------------

    @app.get("/media", response_model=list[MediaOut], tags=["media"])
    async def list_media(
        engine: EngineDep,
        status_filter: Annotated[MediaStatus | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[MediaOut]:
        items = await engine.list_media(status=status_filter, limit=limit, offset=offset)
        return [MediaOut.from_item(i) for i in items]

    @app.get("/media/{media_id}", response_model=MediaDetailOut, tags=["media"])
    async def get_media(media_id: str, engine: EngineDep) -> MediaDetailOut:
        item = await engine.get(media_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"no media with id {media_id}")
        description = await _thread(engine.store.get_description, media_id)
        return MediaDetailOut(
            media=MediaOut.from_item(item),
            description=(
                DescriptionOut(
                    narrative=description.narrative,
                    on_screen_text=description.on_screen_text,
                    tags=description.tags,
                    frame_notes=[n.model_dump() for n in description.frame_notes],
                    confidence=str(description.confidence),
                )
                if description
                else None
            ),
        )

    @app.get("/media/{media_id}/thumbnail", tags=["media"])
    async def get_thumbnail(
        media_id: str,
        engine: EngineDep,
        animated: Annotated[bool, Query(description="Serve the animated preview.")] = False,
    ) -> FileResponse:
        item = await engine.get(media_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"no media with id {media_id}")
        if not item.thumbnail_path:
            raise HTTPException(status_code=404, detail="no thumbnail for this item")

        path = Path(item.thumbnail_path)
        if animated:
            preview = path.with_suffix("").with_suffix(".preview.webp")
            if preview.is_file():
                path = preview
        if not path.is_file():
            raise HTTPException(status_code=404, detail="thumbnail file is missing on disk")
        return FileResponse(path, media_type="image/webp")

    @app.post("/media/{media_id}/reindex", response_model=MediaOut, tags=["media"])
    async def reindex(
        media_id: str,
        engine: EngineDep,
        redescribe: Annotated[
            bool,
            Query(
                description=(
                    "Re-run the vision model. False re-embeds from the stored "
                    "description, which costs nothing in vision spend."
                )
            ),
        ] = True,
    ) -> MediaOut:
        if await engine.get(media_id) is None:
            raise HTTPException(status_code=404, detail=f"no media with id {media_id}")
        item = await engine.reindex(media_id, redescribe=redescribe)
        if item is None:  # pragma: no cover - deleted mid-request
            raise HTTPException(status_code=404, detail=f"no media with id {media_id}")
        return MediaOut.from_item(item)

    @app.delete("/media/{media_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["media"])
    async def delete_media(media_id: str, engine: EngineDep) -> None:
        if not await _thread(engine.store.delete_media, media_id):
            raise HTTPException(status_code=404, detail=f"no media with id {media_id}")

    # -- jobs --------------------------------------------------------------

    @app.get("/jobs", response_model=list[JobOut], tags=["jobs"])
    async def list_jobs(
        engine: EngineDep,
        state: Annotated[str | None, Query(description="pending|running|done|failed")] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[JobOut]:
        jobs = await _thread(engine.store.list_jobs, state=state, limit=limit)
        return [
            JobOut(
                id=j.id,
                media_id=j.media_id,
                kind=str(j.kind),
                state=str(j.state),
                attempts=j.attempts,
                last_error=j.last_error,
                updated_at=j.updated_at,
            )
            for j in jobs
        ]

    return app


async def _thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run blocking storage work off the event loop.

    SQLite calls are synchronous; running them inline would stall every other
    in-flight request on the single event loop thread.
    """
    import asyncio
    import functools

    return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
