"""Deterministic stub provider.

The main regression gate. It lets the whole pipeline — ingest, describe,
chunk, embed, index, search — run in CI with no API key, no network, and no
cost, which is what makes it practical to test the parts that actually break:
resumability, transaction boundaries, ranking, and error handling.

Output is derived from frame bytes, so the same input always yields the same
description and search assertions stay stable across runs. Failures can be
injected to exercise the retry taxonomy without waiting for a real outage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from ...config import VisionSettings
from ...models import Confidence, Description, FrameNote, FrameSample, UsageRecord
from ..base import VisionProvider, VisionRequest

# Deliberately concrete and varied: search tests need descriptions that are
# actually distinguishable from one another, not lorem ipsum.
_SUBJECTS = [
    "a man in a grey suit", "a small dog", "a woman with red hair",
    "a tabby cat", "a group of teenagers", "an elderly man in a flat cap",
    "a parrot", "a chef in whites",
]
_ACTIONS = [
    "slowly turns to face the camera", "knocks a glass off the table",
    "walks out of frame without looking back", "throws both hands in the air",
    "stares blankly then blinks", "collapses onto a sofa",
    "spins around once", "shakes their head and sighs",
]
_SETTINGS = [
    "in a cluttered office", "in a bright kitchen", "on a rainy street",
    "in a hotel lobby", "in a parked car", "on a football pitch",
    "in a lecture hall", "backstage at a concert",
]
_MOODS = ["unimpressed", "delighted", "resigned", "smug", "panicked", "bored"]
_CAPTIONS = ["NOPE", "not today", "WHY", "same", "", "it's fine", "", "ABSOLUTELY NOT"]


@dataclass
class StubVisionProvider(VisionProvider):
    """A vision provider that invents plausible, deterministic descriptions."""

    settings: VisionSettings | None = None
    name: str = "stub"
    model: str = "stub-vision-1"

    fail_with: Exception | None = None
    """Exception to raise instead of answering. Injects provider failures
    without a network."""

    fail_times: int | None = None
    """How many calls fail before one succeeds.

    ``None`` means every call fails — a permanent condition such as a refusal.
    An integer means fail that many times then recover, which is what a
    transient rate limit looks like. ``None`` rather than ``0`` as the
    "always" sentinel so that a counter reaching zero is unambiguously
    *exhausted* rather than *infinite*.
    """

    calls: list[VisionRequest] = field(default_factory=list)
    """Every request received, for asserting on call counts and prompt content."""

    def __post_init__(self) -> None:
        if self.settings is not None:
            self.model = self.settings.model or self.model

    def _maybe_fail(self) -> None:
        if self.fail_with is None:
            return
        if self.fail_times is None:
            raise self.fail_with
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with

    @staticmethod
    def _seed(request: VisionRequest) -> int:
        """Stable seed from frame content, falling back to the instruction."""
        digest = hashlib.sha256()
        for frame in request.frames:
            digest.update(frame.image_bytes)
        if not request.frames:
            digest.update(request.instruction.encode())
        return int.from_bytes(digest.digest()[:8], "big")

    def _compose(self, request: VisionRequest) -> Description:
        seed = self._seed(request)
        subject = _SUBJECTS[seed % len(_SUBJECTS)]
        action = _ACTIONS[(seed >> 8) % len(_ACTIONS)]
        setting = _SETTINGS[(seed >> 16) % len(_SETTINGS)]
        mood = _MOODS[(seed >> 24) % len(_MOODS)]
        caption = _CAPTIONS[(seed >> 32) % len(_CAPTIONS)]

        frames: Sequence[FrameSample] = request.frames
        notes = [
            FrameNote(
                t_ms=frame.t_ms,
                note=(
                    f"{subject.capitalize()} {setting}, looking {mood}."
                    if i == 0
                    else f"{subject.capitalize()} {action}."
                ),
            )
            for i, frame in enumerate(frames)
        ]

        return Description(
            narrative=(
                f"{subject.capitalize()} {setting}. Over the course of the clip "
                f"{subject} {action}, looking {mood} throughout."
            ),
            on_screen_text=caption,
            tags=[
                subject.split()[-1], action.split()[0], setting.split()[-1], mood, "stub",
            ],
            frame_notes=notes,
            confidence=Confidence.HIGH if len(frames) >= 3 else Confidence.MEDIUM,
        )

    async def describe(self, request: VisionRequest) -> tuple[Description, UsageRecord]:
        self.calls.append(request)
        self._maybe_fail()
        description = self._compose(request)
        return description, self._usage(request, len(description.narrative))

    async def complete(self, request: VisionRequest) -> tuple[str, UsageRecord]:
        self.calls.append(request)
        self._maybe_fail()
        if not request.frames:
            # Text-only: the reranker's path. A midpoint score keeps rerank
            # tests about plumbing rather than about stub scoring behaviour.
            return "5", self._usage(request, 1)
        description = self._compose(request)
        text = description.frame_notes[0].note if description.frame_notes else description.narrative
        return text, self._usage(request, len(text))

    def _usage(self, request: VisionRequest, output_chars: int) -> UsageRecord:
        # Roughly four characters per token, plus the w*h/750 image rule, so
        # cost-estimator tests operate on numbers of a believable magnitude.
        image_tokens = sum(max(1, (f.width * f.height) // 750) for f in request.frames)
        return UsageRecord(
            input_tokens=image_tokens + len(request.instruction) // 4 + len(request.system) // 4,
            output_tokens=max(1, output_chars // 4),
            model=self.model,
        )
