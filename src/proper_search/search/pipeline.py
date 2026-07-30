"""The search pipeline: filter, retrieve on several signals, fuse, rerank.

Three retrieval signals run against the same filtered candidate pool:

* **media lexical** — BM25 over narrative, on-screen text, and tags, with
  on-screen text weighted up. Catches exact words: a title, a name, a caption
  someone remembers verbatim.
* **chunk lexical** — BM25 over individual chunks. Catches a keyword that
  appears in one moment but is diluted across the whole description.
* **dense** — vector KNN over chunks, collapsed to each clip's best chunk.
  Catches paraphrase, which is most of what people actually type.

Either family alone has a bad failure mode. Lexical returns nothing for "the
guy who looks completely done with everything" because none of those words are
in the description. Dense quietly misses an exact rare token like a name,
because one unusual word barely moves a sentence embedding. Fusing them covers
both, and RRF makes it possible without calibrating scores against each other.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from ..config import SearchSettings
from ..embed.base import EmbeddingProvider, InputType
from ..errors import ProviderError, StorageError
from ..models import ChunkKind, SearchFilters, SearchHit
from ..store.base import DenseMatch, LexicalMatch, StorageBackend
from ..vision.base import VisionProvider, VisionRequest
from ..vision.prompts import rerank_instruction, rerank_system
from .fusion import FusedResult, collapse_to_best, reciprocal_rank_fusion

SIGNAL_MEDIA_LEXICAL = "bm25_media"
SIGNAL_CHUNK_LEXICAL = "bm25_chunk"
SIGNAL_DENSE = "dense"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(slots=True)
class SearchDiagnostics:
    """What each signal contributed. Returned alongside results for debugging
    relevance, which is otherwise guesswork without a UI."""

    query: str
    candidates_by_signal: dict[str, int]
    fused_count: int
    reranked: bool = False
    degraded_signals: dict[str, str] | None = None


class SearchPipeline:
    """Runs a query across every enabled signal and fuses the results."""

    def __init__(
        self,
        store: StorageBackend,
        embedder: EmbeddingProvider | None,
        settings: SearchSettings,
        *,
        reranker: VisionProvider | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: SearchFilters | None = None,
        explain: bool = False,
    ) -> tuple[list[SearchHit], SearchDiagnostics]:
        limit = limit or self.settings.default_limit
        depth = max(self.settings.candidate_limit, limit)
        query = query.strip()
        if not query:
            return [], SearchDiagnostics(query=query, candidates_by_signal={}, fused_count=0)

        rankings: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        degraded: dict[str, str] = {}
        # chunk_id per media, so a hit can report which moment matched.
        best_chunk: dict[str, int] = {}
        snippets: dict[str, str] = {}

        lexical_task = (
            asyncio.to_thread(self._lexical, query, depth, filters)
            if self.settings.enable_lexical
            else None
        )
        dense_task = (
            self._dense(query, depth, filters)
            if self.settings.enable_dense and self.embedder is not None
            else None
        )

        # Both legs run concurrently: the dense leg waits on a network call to
        # embed the query, and there is no reason for BM25 to sit behind it.
        lexical_result, dense_result = await asyncio.gather(
            lexical_task or _none(), dense_task or _none(), return_exceptions=True
        )

        if isinstance(lexical_result, BaseException):
            if not isinstance(lexical_result, StorageError):
                raise lexical_result
            degraded[SIGNAL_MEDIA_LEXICAL] = str(lexical_result)
        elif lexical_result is not None:
            media_hits, chunk_hits = lexical_result
            rankings[SIGNAL_MEDIA_LEXICAL] = [m.media_id for m in media_hits]
            counts[SIGNAL_MEDIA_LEXICAL] = len(media_hits)
            for match in media_hits:
                snippets.setdefault(match.media_id, match.snippet)
            rankings[SIGNAL_CHUNK_LEXICAL] = collapse_to_best(
                [(m.media_id, m.score) for m in chunk_hits]
            )
            counts[SIGNAL_CHUNK_LEXICAL] = len(chunk_hits)
            for match in chunk_hits:
                if match.chunk_id is not None:
                    best_chunk.setdefault(match.media_id, match.chunk_id)
                snippets.setdefault(match.media_id, match.snippet)

        if isinstance(dense_result, BaseException):
            # A dense outage degrades to lexical-only rather than failing the
            # search. Half a result set beats an error page.
            if not isinstance(dense_result, ProviderError | StorageError):
                raise dense_result
            degraded[SIGNAL_DENSE] = str(dense_result)
        elif dense_result is not None:
            rankings[SIGNAL_DENSE] = collapse_to_best(
                [(m.media_id, m.distance) for m in dense_result], higher_is_better=False
            )
            counts[SIGNAL_DENSE] = len(dense_result)
            for hit in dense_result:
                best_chunk.setdefault(hit.media_id, hit.chunk_id)

        if not rankings:
            return [], SearchDiagnostics(
                query=query,
                candidates_by_signal=counts,
                fused_count=0,
                degraded_signals=degraded or None,
            )

        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)

        if self.settings.rerank == "llm" and self.reranker is not None:
            fused = await self._rerank(query, fused)

        hits = await asyncio.to_thread(
            self._hydrate, fused[:limit], best_chunk, snippets, explain
        )
        return hits, SearchDiagnostics(
            query=query,
            candidates_by_signal=counts,
            fused_count=len(fused),
            reranked=self.settings.rerank == "llm" and self.reranker is not None,
            degraded_signals=degraded or None,
        )

    # -- signals -----------------------------------------------------------

    def _lexical(
        self, query: str, depth: int, filters: SearchFilters | None
    ) -> tuple[list[LexicalMatch], list[LexicalMatch]]:
        weights = (
            self.settings.weight_narrative,
            self.settings.weight_on_screen_text,
            self.settings.weight_tags,
        )
        media_hits = self.store.search_lexical(
            query, limit=depth, filters=filters, weights=weights
        )
        chunk_hits = self.store.search_chunks_lexical(query, limit=depth, filters=filters)
        return media_hits, chunk_hits

    async def _dense(
        self, query: str, depth: int, filters: SearchFilters | None
    ) -> list[DenseMatch]:
        embedder = self.embedder
        assert embedder is not None
        vector = await embedder.embed_one(query, input_type=InputType.QUERY)
        return await asyncio.to_thread(
            self.store.search_dense, vector, limit=depth, filters=filters
        )

    # -- rerank ------------------------------------------------------------

    async def _rerank(self, query: str, fused: list[FusedResult]) -> list[FusedResult]:
        """Re-score the top candidates with a model.

        Only the head of the list is rescored — reranking is a per-item model
        call, so doing it over the full candidate pool would cost more than the
        original search by an order of magnitude for no benefit past the point
        where results stop being plausible.
        """
        reranker = self.reranker
        assert reranker is not None
        head = fused[: self.settings.rerank_top_n]
        tail = fused[self.settings.rerank_top_n :]
        if not head:
            return fused

        items = await asyncio.to_thread(
            self.store.get_media_many, [f.key for f in head]
        )

        async def score(entry: FusedResult) -> tuple[str, float]:
            media = items.get(entry.key)
            if media is None:
                return entry.key, 0.0
            description = await asyncio.to_thread(self.store.get_description, entry.key)
            if description is None:
                return entry.key, 0.0
            try:
                text, _ = await reranker.complete(
                    VisionRequest(
                        system=rerank_system(),
                        instruction=rerank_instruction(
                            query, description.narrative, description.on_screen_text
                        ),
                        max_tokens=8,
                    )
                )
            except ProviderError:
                # A reranker failure must not lose the result; fall back to its
                # fused position by scoring it neutrally.
                return entry.key, 5.0
            match = _NUMBER.search(text)
            return entry.key, float(match.group()) if match else 5.0

        scored = dict(await asyncio.gather(*(score(entry) for entry in head)))

        for entry in head:
            entry.ranks["rerank"] = int(scored.get(entry.key, 5.0))
        head.sort(key=lambda e: (-scored.get(e.key, 0.0), e.key))
        # Reranked head keeps its order ahead of the untouched tail.
        return head + tail

    # -- hydration ---------------------------------------------------------

    def _hydrate(
        self,
        fused: list[FusedResult],
        best_chunk: dict[str, int],
        snippets: dict[str, str],
        explain: bool,
    ) -> list[SearchHit]:
        if not fused:
            return []
        items = self.store.get_media_many([entry.key for entry in fused])
        hits: list[SearchHit] = []
        for entry in fused:
            media = items.get(entry.key)
            if media is None:
                # Deleted between retrieval and hydration; skip rather than
                # returning a hit that cannot be opened.
                continue
            snippet = snippets.get(entry.key, "")
            t_ms: int | None = None
            kind: ChunkKind | None = None
            chunk_id = best_chunk.get(entry.key)
            if chunk_id is not None:
                chunk = self.store.get_chunk(chunk_id)
                if chunk is not None:
                    kind = chunk.kind
                    t_ms = chunk.t_start_ms
                    if not snippet:
                        snippet = chunk.text
            hits.append(
                SearchHit(
                    media=media,
                    score=entry.score,
                    signals={k: float(v) for k, v in entry.ranks.items()} if explain else {},
                    snippet=snippet,
                    snippet_t_ms=t_ms,
                    matched_chunk_kind=kind,
                )
            )
        return hits


async def _none() -> None:
    """Placeholder for a disabled signal in the concurrent gather."""
    return None
