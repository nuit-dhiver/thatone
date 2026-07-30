"""Cost estimation.

A 100k-item backfill is a real bill, and the difference between the cheap and
expensive configurations is roughly tenfold. Committing to that without a
number first is how people get surprised, so this samples a handful of actual
items, measures what they would really cost, and projects.

Two accuracy choices matter:

* **Measure, don't model.** Frames are sampled for real and, where the provider
  offers a token counter, counted for real. Image tokens do not follow a clean
  formula across resolutions, so an analytical estimate drifts exactly where
  the money is.
* **Sample the corpus, don't average the config.** Per-item cost depends on
  clip length and how much motion it contains, which vary wildly. Sampling real
  items captures that; multiplying a nominal frame count does not.

Rates come from config rather than being hardcoded, so a published price change
is an edit to a YAML file, not a release.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .media.sampling import sample_frames
from .models import FrameSample
from .vision import prompts
from .vision.base import VisionProvider, VisionRequest, estimate_image_tokens

CHARS_PER_TOKEN = 4
"""Crude but adequate for text, which is a small share of the total next to
images. Only used when the provider offers no token counter."""


@dataclass
class ItemEstimate:
    path: str
    frames: int
    input_tokens: int
    output_tokens: int
    measured: bool = False
    """True when the provider counted the tokens, False when approximated."""


@dataclass
class CostEstimate:
    """Projection for a full run."""

    item_count: int
    sampled: int
    model: str
    strategy: str
    batch: bool

    avg_frames: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    vision_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0

    measured: bool = False
    warnings: list[str] = field(default_factory=list)
    samples: list[ItemEstimate] = field(default_factory=list)

    @property
    def cost_per_item(self) -> float:
        return self.total_cost / self.item_count if self.item_count else 0.0

    def summary(self) -> str:
        basis = "measured" if self.measured else "approximated"
        return (
            f"{self.item_count} items x ${self.cost_per_item:.5f} = ${self.total_cost:.2f} "
            f"({self.model}, {self.strategy}, {basis}"
            f"{', batch' if self.batch else ''})"
        )


EXPECTED_OUTPUT_TOKENS = 400
"""A description with per-frame notes plus a narrative lands near here. Only a
fallback: the sampled items' real output length replaces it when available."""


class CostEstimator:
    """Projects the cost of describing a corpus."""

    def __init__(self, settings: Settings, vision: VisionProvider | None = None) -> None:
        self.settings = settings
        self.vision = vision

    async def estimate_paths(
        self,
        paths: list[str | Path],
        *,
        sample_size: int = 5,
        seed: int | None = None,
    ) -> CostEstimate:
        """Sample from ``paths`` and project the cost of describing all of them."""
        total = len(paths)
        estimate = CostEstimate(
            item_count=total,
            sampled=0,
            model=self.settings.vision.model,
            strategy=self.settings.vision.strategy,
            batch=self.settings.vision.use_batch_api,
        )
        if total == 0:
            return estimate

        rng = random.Random(seed)
        chosen = rng.sample(paths, min(sample_size, total))

        for path in chosen:
            try:
                item = await self._estimate_one(path)
            except Exception as exc:
                estimate.warnings.append(f"skipped {path}: {exc}")
                continue
            estimate.samples.append(item)

        if not estimate.samples:
            estimate.warnings.append("no sampled item could be measured; no estimate produced")
            return estimate

        estimate.sampled = len(estimate.samples)
        estimate.measured = all(s.measured for s in estimate.samples)
        estimate.avg_frames = sum(s.frames for s in estimate.samples) / estimate.sampled
        estimate.avg_input_tokens = (
            sum(s.input_tokens for s in estimate.samples) / estimate.sampled
        )
        estimate.avg_output_tokens = (
            sum(s.output_tokens for s in estimate.samples) / estimate.sampled
        )

        self._price(estimate)
        return estimate

    async def _estimate_one(self, path: str | Path) -> ItemEstimate:
        frames = await asyncio.to_thread(sample_frames, path, self.settings.sampling)
        request = self._request_for(frames)

        input_tokens, measured = await self._count_input_tokens(request, frames)
        return ItemEstimate(
            path=str(path),
            frames=len(frames),
            input_tokens=input_tokens,
            output_tokens=EXPECTED_OUTPUT_TOKENS,
            measured=measured,
        )

    def _request_for(self, frames: list[FrameSample]) -> VisionRequest:
        duration = frames[-1].t_ms if frames else 0
        return VisionRequest(
            system=prompts.SYSTEM,
            instruction=prompts.single_call_instruction(frames, duration_ms=duration),
            frames=frames,
        )

    async def _count_input_tokens(
        self, request: VisionRequest, frames: list[FrameSample]
    ) -> tuple[int, bool]:
        counter = getattr(self.vision, "count_tokens", None)
        if counter is not None:
            try:
                return int(await counter(request)), True
            except Exception:
                pass
        approximate = (
            estimate_image_tokens(frames)
            + len(request.system) // CHARS_PER_TOKEN
            + len(request.instruction) // CHARS_PER_TOKEN
        )
        return approximate, False

    def _price(self, estimate: CostEstimate) -> None:
        pricing = self.settings.pricing
        price = pricing.for_model(self.settings.vision.model)
        if price is None:
            estimate.warnings.append(
                f"no price configured for {self.settings.vision.model!r}; set "
                f"pricing.models['{self.settings.vision.model}'] to get a cost figure"
            )
            return

        strategy = self.settings.vision.strategy
        # Cost scales with request count, not just tokens: sequential issues one
        # request per frame, two_pass one per frame plus a synthesis.
        if strategy == "sequential":
            requests_per_item = estimate.avg_frames
        elif strategy == "two_pass":
            requests_per_item = estimate.avg_frames + 1
        else:
            requests_per_item = 1.0

        discount = pricing.batch_discount if self.settings.vision.use_batch_api else 1.0

        per_item = price.cost(
            input_tokens=int(estimate.avg_input_tokens * requests_per_item),
            output_tokens=int(estimate.avg_output_tokens * requests_per_item),
            batch_discount=discount,
        )
        estimate.vision_cost = per_item * estimate.item_count

        # Embedding is a rounding error next to vision, but reporting zero
        # would imply it is free and invite surprise on a large corpus.
        chunks_per_item = estimate.avg_frames + 2
        embed_tokens = chunks_per_item * 30
        estimate.embedding_cost = (
            embed_tokens * estimate.item_count * pricing.embedding_per_mtok / 1_000_000
        )

        estimate.total_cost = estimate.vision_cost + estimate.embedding_cost

        if not estimate.measured:
            estimate.warnings.append(
                "token counts are approximated because the provider exposes no counter; "
                "treat this as a rough figure"
            )
        if self.settings.vision.use_batch_api:
            estimate.warnings.append(
                f"assumes the batch discount of {pricing.batch_discount:.0%} applies"
            )
