"""Hybrid search: lexical + dense retrieval, fused by rank."""

from .fusion import FusedResult, collapse_to_best, reciprocal_rank_fusion
from .pipeline import (
    SIGNAL_CHUNK_LEXICAL,
    SIGNAL_DENSE,
    SIGNAL_MEDIA_LEXICAL,
    SearchDiagnostics,
    SearchPipeline,
)

__all__ = [  # noqa: RUF022
    "SearchPipeline",
    "SearchDiagnostics",
    "reciprocal_rank_fusion",
    "collapse_to_best",
    "FusedResult",
    "SIGNAL_MEDIA_LEXICAL",
    "SIGNAL_CHUNK_LEXICAL",
    "SIGNAL_DENSE",
]
