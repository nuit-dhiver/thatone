"""How many model calls to make, and what to ask on each.

Three shapes, same interface, chosen by config. The trade is cost against
temporal fidelity, and at 100k items the difference is the difference between a
$500 run and a $4,000 one.

The default is :class:`SingleCallStrategy` because it gets the "revise the
story as later frames reveal more" behaviour for one request rather than N:
sending every frame at once means the model never commits to a reading of
frame 1 before it has seen frame 8, so there is no early mistake to walk back.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..errors import ProviderResponseInvalid
from ..models import DescribeResult, Description, FrameSample, UsageRecord
from . import prompts
from .base import VisionProvider, VisionRequest


def _render_running(description: Description) -> str:
    """Render a description back into prose for the next sequential turn."""
    lines = [f"Narrative: {description.narrative}"]
    if description.on_screen_text:
        lines.append(f"On-screen text: {description.on_screen_text}")
    if description.tags:
        lines.append(f"Tags: {', '.join(description.tags)}")
    if description.frame_notes:
        lines.append("Frame notes:")
        lines.extend(f"  t={n.t_ms}ms: {n.note}" for n in description.frame_notes)
    return "\n".join(lines)


class DescriptionStrategy(ABC):
    """Orchestrates the provider calls that produce one description."""

    name: str

    @abstractmethod
    async def run(
        self,
        provider: VisionProvider,
        frames: Sequence[FrameSample],
        *,
        duration_ms: int,
        caption_provider: VisionProvider | None = None,
    ) -> DescribeResult: ...

    @staticmethod
    def _require_frames(frames: Sequence[FrameSample]) -> None:
        if not frames:
            raise ProviderResponseInvalid("cannot describe an item with no sampled frames")


class SingleCallStrategy(DescriptionStrategy):
    """All frames in one request. The default.

    One request per item, so cost scales with frame count only through image
    tokens rather than through request count. The model sees the full sequence
    before writing, which is where cross-frame revision comes from.
    """

    name = "single_call"

    async def run(
        self,
        provider: VisionProvider,
        frames: Sequence[FrameSample],
        *,
        duration_ms: int,
        caption_provider: VisionProvider | None = None,
    ) -> DescribeResult:
        self._require_frames(frames)
        request = VisionRequest(
            system=prompts.SYSTEM,
            instruction=prompts.single_call_instruction(frames, duration_ms=duration_ms),
            frames=frames,
        )
        description, usage = await provider.describe(request)
        return DescribeResult(
            description=description, usage=usage, model=provider.model, strategy=self.name
        )


class SequentialStrategy(DescriptionStrategy):
    """One request per frame, each revising the running description.

    The highest-fidelity option for subtle temporal detail, and the most
    expensive: N requests per item, each re-sending the accumulated
    description. At 100k items with 8 frames that is 800k requests, so it is
    worth reserving for a subset rather than a full corpus.

    The system prompt is identical on every turn, so prompt caching absorbs a
    meaningful share of the repeated cost.
    """

    name = "sequential"

    async def run(
        self,
        provider: VisionProvider,
        frames: Sequence[FrameSample],
        *,
        duration_ms: int,
        caption_provider: VisionProvider | None = None,
    ) -> DescribeResult:
        self._require_frames(frames)
        total = len(frames)
        total_usage = UsageRecord()

        description, usage = await provider.describe(
            VisionRequest(
                system=prompts.SYSTEM,
                instruction=prompts.sequential_first_instruction(
                    frames[0], duration_ms=duration_ms, total=total
                ),
                frames=[frames[0]],
            )
        )
        total_usage = total_usage + usage

        for position, frame in enumerate(frames[1:], start=2):
            description, usage = await provider.describe(
                VisionRequest(
                    system=prompts.SYSTEM,
                    instruction=prompts.sequential_next_instruction(
                        frame,
                        position=position,
                        total=total,
                        running=_render_running(description),
                    ),
                    frames=[frame],
                )
            )
            total_usage = total_usage + usage

        return DescribeResult(
            description=description,
            usage=total_usage,
            model=provider.model,
            strategy=self.name,
        )


class TwoPassStrategy(DescriptionStrategy):
    """Cheap parallel captions, then one strong-model synthesis.

    The captions are independent, so they run concurrently and can use a small
    model; only the synthesis needs to reason across time. Cost lands between
    the other two, and the failure mode is specific: a caption written without
    temporal context misreads things the sequence makes obvious, so the
    synthesis prompt is explicit that the sequence wins.
    """

    name = "two_pass"

    async def run(
        self,
        provider: VisionProvider,
        frames: Sequence[FrameSample],
        *,
        duration_ms: int,
        caption_provider: VisionProvider | None = None,
    ) -> DescribeResult:
        self._require_frames(frames)
        captioner = caption_provider or provider
        total_usage = UsageRecord()

        async def caption(frame: FrameSample) -> tuple[int, str, UsageRecord]:
            text, usage = await captioner.complete(
                VisionRequest(
                    system=prompts.SYSTEM,
                    instruction=prompts.caption_instruction(frame),
                    frames=[frame],
                )
            )
            return frame.t_ms, text, usage

        results = await asyncio.gather(*(caption(frame) for frame in frames))
        captions: list[tuple[int, str]] = []
        for t_ms, text, usage in results:
            captions.append((t_ms, text))
            total_usage = total_usage + usage

        description, usage = await provider.describe(
            VisionRequest(
                system=prompts.SYSTEM,
                instruction=prompts.synthesis_instruction(captions, duration_ms=duration_ms),
                # No frames: the synthesis pass reasons over the captions. Re-sending
                # the images here would double the image-token cost for the pass whose
                # whole point is that it does not need to look again.
                frames=(),
            )
        )
        total_usage = total_usage + usage

        return DescribeResult(
            description=description,
            usage=total_usage,
            model=provider.model,
            strategy=self.name,
        )


_STRATEGIES: dict[str, DescriptionStrategy] = {
    s.name: s for s in (SingleCallStrategy(), SequentialStrategy(), TwoPassStrategy())
}


def get_strategy(name: str) -> DescriptionStrategy:
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown description strategy {name!r}; available: {sorted(_STRATEGIES)}"
        ) from None
