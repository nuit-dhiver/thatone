"""Vision layer: turn sampled frames into a searchable description."""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import VisionProvider, VisionRequest, estimate_image_tokens
from .prompts import PROMPT_VERSION
from .providers.stub import StubVisionProvider
from .schema import to_api_schema
from .strategies import (
    DescriptionStrategy,
    SequentialStrategy,
    SingleCallStrategy,
    TwoPassStrategy,
    get_strategy,
)

__all__ = [  # noqa: RUF022
    "VisionProvider",
    "VisionRequest",
    "estimate_image_tokens",
    "PROMPT_VERSION",
    "to_api_schema",
    "DescriptionStrategy",
    "SingleCallStrategy",
    "SequentialStrategy",
    "TwoPassStrategy",
    "get_strategy",
    "StubVisionProvider",
    "build_provider",
    "build_caption_provider",
]


def build_provider(settings: Settings) -> VisionProvider:
    """Construct the configured vision provider."""
    vision = settings.vision
    if vision.provider == "stub":
        return StubVisionProvider(settings=vision)
    if vision.provider == "anthropic":
        from .providers.anthropic import AnthropicVisionProvider

        return AnthropicVisionProvider(vision, settings.require_api_key(vision.api_key_env))
    if vision.provider in ("gemini", "openai_compat"):
        raise ConfigError(
            f"the {vision.provider!r} vision provider is not implemented yet; "
            f"use vision.provider: anthropic"
        )
    raise ConfigError(f"unknown vision provider: {vision.provider!r}")


def build_caption_provider(settings: Settings) -> VisionProvider | None:
    """Provider for the two_pass caption stage, if a cheaper model is configured."""
    caption_model = settings.vision.caption_model
    if not caption_model or caption_model == settings.vision.model:
        return None
    scoped = settings.model_copy(deep=True)
    scoped.vision.model = caption_model
    return build_provider(scoped)
