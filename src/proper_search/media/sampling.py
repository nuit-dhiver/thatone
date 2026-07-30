"""Frame sampling strategies.

Which frames get sent to the vision model is the single biggest lever on both
cost and description quality, so the choice is pluggable and the default is
opinionated.

All strategies run against a *timeline* — one lightweight record per frame,
pixels already discarded — rather than against the file. That means:

* every strategy sees true per-frame timestamps, which matters because GIF
  frame delays vary and an ``index / fps`` estimate drifts badly;
* strategy logic is pure and unit-testable without any media on disk.

The cost is decoding twice: once to build the timeline, once to extract the
chosen frames. For GIFs and short clips that is negligible next to the vision
call that follows, and it buys uniform, globally-correct selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import SamplingSettings
from ..errors import DecodeError
from ..models import FrameSample, MediaProbe
from .decode import DecodedFrame, decode_at_indices, encode_frame, iter_frames
from .hashing import dhash, hamming


@dataclass(slots=True, frozen=True)
class FrameMeta:
    """One frame's position and appearance, without its pixels."""

    index: int
    t_ms: int
    phash: int


def scan_timeline(path: str | Path, *, max_decode_frames: int = 3000) -> list[FrameMeta]:
    """Decode once to build the frame timeline, discarding pixels as we go.

    Memory stays flat regardless of clip length — a few dozen bytes per frame —
    which is what makes it safe to run over an untrusted corpus.
    """
    timeline: list[FrameMeta] = []
    for frame in iter_frames(path, max_frames=max_decode_frames):
        timeline.append(
            FrameMeta(index=frame.index, t_ms=frame.t_ms, phash=dhash(frame.image))
        )
    if not timeline:
        raise DecodeError(f"{path} decoded zero frames")
    return timeline


def _evenly_spaced(items: Sequence[FrameMeta], count: int) -> list[FrameMeta]:
    """Pick ``count`` items spread across ``items``, always including the first.

    Used both to thin an over-long selection and to top up a sparse one.
    """
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


class SamplingStrategy(ABC):
    """Chooses which frames to describe."""

    name: str

    @abstractmethod
    def select(self, timeline: Sequence[FrameMeta], settings: SamplingSettings) -> list[FrameMeta]:
        """Return the chosen frames, in time order."""

    @staticmethod
    def _cap(frames: Sequence[FrameMeta], settings: SamplingSettings) -> list[FrameMeta]:
        """Enforce ``max_frames`` as a hard cost ceiling for every strategy."""
        if len(frames) > settings.max_frames:
            return _evenly_spaced(frames, settings.max_frames)
        return list(frames)


class AdaptiveSampler(SamplingStrategy):
    """Keep a frame when the picture has actually changed. The default.

    Fixed-interval sampling is a poor fit for this corpus: most GIFs run 1-4
    seconds, so "every 2 seconds" yields a single frame and the model never
    sees the thing that happens. Meanwhile a 60-second mostly-static clip
    yields 30 near-identical frames and you pay for all of them.

    Scene-change detection sizes the sample to the content instead — a
    12-frame reaction GIF gives ~3 distinct moments, a long clip gives its
    actual beats.
    """

    name = "adaptive"

    def select(self, timeline: Sequence[FrameMeta], settings: SamplingSettings) -> list[FrameMeta]:
        if not timeline:
            return []

        kept = [timeline[0]]
        for meta in timeline[1:]:
            if hamming(meta.phash, kept[-1].phash) >= settings.hamming_threshold:
                kept.append(meta)

        # A static or near-static clip trips the threshold rarely or never.
        # Top up from the full timeline so the model still gets temporal
        # context — a slow zoom is not the same as a freeze frame, and one
        # frame cannot tell the model which it is.
        if len(kept) < settings.min_frames:
            target = min(settings.min_frames, len(timeline))
            merged = {m.index: m for m in kept}
            for meta in _evenly_spaced(timeline, target):
                merged[meta.index] = meta
            kept = [merged[i] for i in sorted(merged)]

        return self._cap(kept, settings)


class IntervalSampler(SamplingStrategy):
    """One frame per fixed time interval.

    Predictable, and a poor fit for short GIFs — kept because it is the right
    choice for long-form video where wall-clock coverage is what matters.
    """

    name = "interval"

    def select(self, timeline: Sequence[FrameMeta], settings: SamplingSettings) -> list[FrameMeta]:
        if not timeline:
            return []
        step_ms = int(settings.interval_seconds * 1000)
        if step_ms <= 0:
            return self._cap(timeline, settings)

        picked: list[FrameMeta] = []
        next_target = timeline[0].t_ms
        for meta in timeline:
            if meta.t_ms >= next_target:
                picked.append(meta)
                next_target = meta.t_ms + step_ms
        if not picked:
            picked = [timeline[0]]
        return self._cap(picked, settings)


class CountSampler(SamplingStrategy):
    """A fixed number of evenly spaced frames.

    The predictable-cost option: every item bills the same, which makes a
    100k-item backfill straightforward to budget.
    """

    name = "count"

    def select(self, timeline: Sequence[FrameMeta], settings: SamplingSettings) -> list[FrameMeta]:
        if not timeline:
            return []
        target = min(settings.target_count, settings.max_frames)
        return _evenly_spaced(timeline, target)


_STRATEGIES: dict[str, SamplingStrategy] = {
    s.name: s for s in (AdaptiveSampler(), IntervalSampler(), CountSampler())
}


def get_strategy(name: str) -> SamplingStrategy:
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown sampling strategy {name!r}; available: {sorted(_STRATEGIES)}"
        ) from None


def sample_frames(
    path: str | Path,
    settings: SamplingSettings,
    *,
    timeline: Sequence[FrameMeta] | None = None,
) -> list[FrameSample]:
    """Select, decode, and encode the frames to describe.

    ``timeline`` can be supplied by a caller that already scanned the file, so
    ingest does not decode a third time.
    """
    scanned = timeline if timeline is not None else scan_timeline(
        path, max_decode_frames=settings.max_decode_frames
    )
    chosen = get_strategy(settings.strategy).select(scanned, settings)
    if not chosen:
        raise DecodeError(f"sampling selected no frames from {path}")

    by_index = {m.index: m for m in chosen}
    decoded: list[DecodedFrame] = decode_at_indices(
        path, sorted(by_index), max_frames=settings.max_decode_frames
    )

    samples: list[FrameSample] = []
    for frame in sorted(decoded, key=lambda f: f.index):
        payload, width, height = encode_frame(
            frame.image, max_edge=settings.frame_max_edge, quality=settings.jpeg_quality
        )
        samples.append(
            FrameSample(
                index=frame.index,
                t_ms=frame.t_ms,
                image_bytes=payload,
                media_type="image/jpeg",
                width=width,
                height=height,
                phash=by_index[frame.index].phash,
            )
        )
    return samples


def summarize(timeline: Sequence[FrameMeta], probe: MediaProbe) -> str:
    """One-line description of a timeline, for logs and debugging."""
    return (
        f"{len(timeline)} frames over {probe.duration_ms}ms "
        f"({probe.width}x{probe.height}, {probe.mime})"
    )
