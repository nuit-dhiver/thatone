"""Reciprocal rank fusion.

Combining a BM25 list with a vector list is harder than it looks, because their
scores are not comparable and not even on the same kind of scale: BM25 is
unbounded and corpus-dependent, cosine distance is bounded, and normalising
either one makes the result depend on the arbitrary composition of that
particular result set.

RRF sidesteps the problem by discarding scores entirely and fusing on *rank*::

    score(d) = Σ weight_i / (k + rank_i(d))

A document ranked 1st by either signal gets a large contribution; one ranked
50th by both gets a small one. Nothing needs calibrating, adding a third signal
later requires no re-tuning, and a signal that ranks a document nowhere simply
contributes zero rather than dragging the fused score down.

``k`` (default 60, from the original TREC work) sets how sharply rank position
matters: small ``k`` concentrates weight on the very top few, large ``k``
flattens the curve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedResult:
    """One item's fused score, with the per-signal ranks that produced it."""

    key: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    """Signal name to 1-based rank. Kept so a surprising result can be
    explained rather than just observed."""


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedResult]:
    """Fuse named ranked lists into one ordered list.

    ``rankings`` maps a signal name to its ranked keys, best first. Duplicate
    keys within one list are ignored after the first occurrence, so a caller
    that has already collapsed chunk hits to media ids gets the expected
    result.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for signal, keys in rankings.items():
        weight = weights.get(signal, 1.0)
        seen: set[str] = set()
        position = 0
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            position += 1
            scores[key] = scores.get(key, 0.0) + weight / (k + position)
            ranks.setdefault(key, {})[signal] = position

    fused = [FusedResult(key=key, score=score, ranks=ranks[key]) for key, score in scores.items()]
    # Ties broken by key so ordering is deterministic across runs; a wobbling
    # result order makes eval numbers impossible to trust.
    fused.sort(key=lambda item: (-item.score, item.key))
    return fused


def collapse_to_best(
    matches: Sequence[tuple[str, float]], *, higher_is_better: bool = True
) -> list[str]:
    """Reduce per-chunk matches to a ranked list of unique parent keys.

    Max-pooling rather than averaging, and this is the point of chunking at
    all: a clip should rank on its single best-matching moment. Averaging over
    a clip's chunks penalises exactly the case the design targets — a long
    description where one instant matches the query perfectly and the rest is
    unrelated.
    """
    best: dict[str, float] = {}
    for key, value in matches:
        current = best.get(key)
        if current is None:
            best[key] = value
        elif higher_is_better:
            best[key] = max(current, value)
        else:
            best[key] = min(current, value)
    ordered = sorted(best.items(), key=lambda kv: (-kv[1] if higher_is_better else kv[1], kv[0]))
    return [key for key, _ in ordered]
