"""Search-quality evaluation.

There is no UI, so nobody is going to eyeball a result grid and notice that
relevance got worse. This harness is the feedback loop instead: a set of
queries with known answers, run against the index, scored.

It reports each signal in isolation as well as fused, because the interesting
question is never "is search good" but "is fusion actually earning its keep".
If lexical alone matches the fused numbers, the dense leg is costing money for
nothing; if fused is *worse* than either alone, something is wrong with the
weighting rather than with retrieval.

Two metrics, measuring different things:

``recall@k``
    Did the right clip appear in the top k at all? This is what a user with a
    result grid in front of them cares about — they will happily scan ten
    thumbnails.
``MRR``
    How high did it rank? Rewards putting the answer first. Worth watching
    alongside recall, since a change can lift one and flatten the other.

The queries must be written the way people actually recall clips: vague,
partial, and sometimes wrong in their details. A golden set of clean keyword
queries measures nothing this system was built for.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SearchSettings
from ..models import SearchFilters
from ..search.pipeline import SearchPipeline
from ..store.base import StorageBackend

DEFAULT_K_VALUES = (1, 5, 10)


@dataclass(slots=True)
class GoldenQuery:
    """One query and the clip(s) that should answer it."""

    query: str
    expected: list[str]
    """Media ids. Any one of them counts as correct — near-duplicates in a real
    corpus mean several clips can legitimately satisfy one query."""

    note: str = ""
    """Why this query is here. Invaluable when a regression turns up months
    later and nobody remembers what the case was testing."""

    filters: SearchFilters | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenQuery:
        expected = raw.get("expected", raw.get("expected_media_id"))
        if isinstance(expected, str):
            expected = [expected]
        return cls(
            query=raw["query"],
            expected=list(expected or []),
            note=raw.get("note", ""),
        )


@dataclass(slots=True)
class QueryOutcome:
    query: str
    expected: list[str]
    rank: int | None
    """1-based rank of the first correct hit, or None if it never appeared."""

    top_hit: str | None = None
    note: str = ""


@dataclass
class EvalResult:
    """Scores for one retrieval configuration."""

    name: str
    query_count: int
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    outcomes: list[QueryOutcome] = field(default_factory=list)

    @property
    def misses(self) -> list[QueryOutcome]:
        """Queries whose answer never surfaced. The list worth reading."""
        return [o for o in self.outcomes if o.rank is None]

    def format_row(self, k_values: Sequence[int] = DEFAULT_K_VALUES) -> str:
        cells = "  ".join(f"r@{k}={self.recall_at.get(k, 0.0):.2f}" for k in k_values)
        return f"{self.name:<22} {cells}  mrr={self.mrr:.3f}  misses={len(self.misses)}"


def load_golden_set(path: str | Path) -> list[GoldenQuery]:
    """Load a golden set from JSON.

    Expected shape::

        [{"query": "the guy who slowly turns around",
          "expected": ["<media_id>"],
          "note": "vague, no literal keyword overlap"}]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("queries", [])
    return [GoldenQuery.from_dict(item) for item in raw]


def save_golden_set(queries: Sequence[GoldenQuery], path: str | Path) -> None:
    payload = [
        {"query": q.query, "expected": q.expected, "note": q.note} for q in queries
    ]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _configurations(base: SearchSettings) -> dict[str, SearchSettings]:
    """The configurations worth comparing on every run.

    Each isolates one thing, so a number moving points at a cause rather than
    just registering that something changed.
    """
    lexical = base.model_copy(deep=True)
    lexical.enable_dense = False
    lexical.rerank = "none"

    dense = base.model_copy(deep=True)
    dense.enable_lexical = False
    dense.rerank = "none"

    fused = base.model_copy(deep=True)
    fused.rerank = "none"

    return {"lexical only": lexical, "dense only": dense, "fused": fused}


class EvalHarness:
    """Scores a golden set against one or more retrieval configurations."""

    def __init__(
        self,
        store: StorageBackend,
        embedder: object | None,
        settings: SearchSettings,
        *,
        reranker: object | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    async def run(
        self,
        golden: Sequence[GoldenQuery],
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        include_rerank: bool = False,
    ) -> list[EvalResult]:
        """Score every configuration against the golden set."""
        if not golden:
            return []

        configs = _configurations(self.settings)
        if include_rerank and self.reranker is not None:
            with_rerank = self.settings.model_copy(deep=True)
            with_rerank.rerank = "llm"
            configs["fused + rerank"] = with_rerank

        results: list[EvalResult] = []
        for name, config in configs.items():
            if config.enable_dense and self.embedder is None:
                # Skip rather than report a zero, which would read as a
                # quality regression instead of a missing provider.
                continue
            results.append(await self._score(name, config, golden, k_values))
        return results

    async def _score(
        self,
        name: str,
        config: SearchSettings,
        golden: Sequence[GoldenQuery],
        k_values: Sequence[int],
    ) -> EvalResult:
        pipeline = SearchPipeline(
            self.store,
            self.embedder,  # type: ignore[arg-type]
            config,
            reranker=self.reranker,  # type: ignore[arg-type]
        )
        depth = max(k_values)
        outcomes: list[QueryOutcome] = []

        for item in golden:
            hits, _ = await pipeline.search(
                item.query, limit=depth, filters=item.filters
            )
            ids = [h.media.id for h in hits]
            rank = next(
                (i + 1 for i, media_id in enumerate(ids) if media_id in item.expected),
                None,
            )
            outcomes.append(
                QueryOutcome(
                    query=item.query,
                    expected=item.expected,
                    rank=rank,
                    top_hit=ids[0] if ids else None,
                    note=item.note,
                )
            )

        total = len(outcomes)
        recall = {
            k: sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / total
            for k in k_values
        }
        mrr = sum(1.0 / o.rank for o in outcomes if o.rank is not None) / total
        return EvalResult(
            name=name, query_count=total, recall_at=recall, mrr=mrr, outcomes=outcomes
        )


def format_report(
    results: Sequence[EvalResult], *, k_values: Sequence[int] = DEFAULT_K_VALUES
) -> str:
    """Render results as a readable block, with misses listed.

    The misses are the point. Aggregate numbers say whether things moved; the
    individual failures say what to fix.
    """
    if not results:
        return "no results (empty golden set?)"

    lines = [f"search quality over {results[0].query_count} queries", ""]
    lines.extend(r.format_row(k_values) for r in results)

    best = max(results, key=lambda r: r.mrr)
    if best.misses:
        lines.extend(["", f"queries {best.name!r} could not answer:"])
        for outcome in best.misses[:15]:
            suffix = f"  ({outcome.note})" if outcome.note else ""
            lines.append(f"  - {outcome.query!r}{suffix}")
        if len(best.misses) > 15:
            lines.append(f"  ... and {len(best.misses) - 15} more")

    return "\n".join(lines)
