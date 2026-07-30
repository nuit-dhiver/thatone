"""Search-quality evaluation. Without a UI, this is the feedback loop."""

from .harness import (
    DEFAULT_K_VALUES,
    EvalHarness,
    EvalResult,
    GoldenQuery,
    QueryOutcome,
    format_report,
    load_golden_set,
    save_golden_set,
)

__all__ = [  # noqa: RUF022
    "EvalHarness",
    "EvalResult",
    "GoldenQuery",
    "QueryOutcome",
    "load_golden_set",
    "save_golden_set",
    "format_report",
    "DEFAULT_K_VALUES",
]
