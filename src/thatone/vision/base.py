"""Vision provider interface.

Providers are thin: they know how to send frames plus an instruction and get
back either free text or a validated :class:`~thatone.models.Description`.
Everything about *how many* calls to make and *what* to ask lives in
:mod:`thatone.vision.strategies`, so adding a provider never means
reimplementing the description logic.

Every provider is an HTTP client. This package does not host, download, or run
models — a self-hosted model is reached by pointing ``base_url`` at the server
running it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import Description, FrameSample, UsageRecord


@dataclass(slots=True)
class VisionRequest:
    """One call to a vision model.

    ``frames`` may be empty, which makes this a plain text completion — the
    reranker uses that path.
    """

    instruction: str
    system: str = ""
    frames: Sequence[FrameSample] = field(default_factory=tuple)
    max_tokens: int | None = None
    cache_system: bool = True
    """Mark the system prompt as cacheable. Only takes effect above the model's
    minimum cacheable prefix; below it the flag is silently inert."""


class VisionProvider(ABC):
    """Sends frames to a model and returns what it saw."""

    name: str
    model: str

    @abstractmethod
    async def describe(self, request: VisionRequest) -> tuple[Description, UsageRecord]:
        """Return a validated description.

        Implementations must raise
        :class:`~thatone.errors.ProviderRefusal` when the model declines,
        rather than returning an empty description — a refusal is a permanent
        outcome for that item and must not be retried or indexed as content.
        """

    @abstractmethod
    async def complete(self, request: VisionRequest) -> tuple[str, UsageRecord]:
        """Return free-form text. Used for per-frame captions and reranking."""

    async def close(self) -> None:  # noqa: B027
        """Release any underlying HTTP client.

        Deliberately concrete and empty: a provider with nothing to release
        should not be forced to write an empty override.
        """

    async def __aenter__(self) -> VisionProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def estimate_image_tokens(frames: Sequence[FrameSample]) -> int:
    """Approximate image-token cost of a frame set.

    Uses the ``width * height / 750`` rule of thumb. Good enough to reason
    about the cost of a sampling change; the cost estimator uses the provider's
    real token counter for anything that gets reported as a number.
    """
    return sum(max(1, (f.width * f.height) // 750) for f in frames)
