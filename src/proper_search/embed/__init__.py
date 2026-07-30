"""Embedding layer: chunking and vector generation."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..errors import ConfigError
from .base import EmbeddingProvider, HTTPEmbeddingProvider, InputType
from .chunking import build_chunks, estimate_tokens, split_text
from .providers import (
    CohereEmbeddingProvider,
    OpenAIEmbeddingProvider,
    StubEmbeddingProvider,
    VoyageEmbeddingProvider,
)

__all__ = [  # noqa: RUF022
    "EmbeddingProvider",
    "HTTPEmbeddingProvider",
    "InputType",
    "build_chunks",
    "split_text",
    "estimate_tokens",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "CohereEmbeddingProvider",
    "StubEmbeddingProvider",
    "build_embedding_provider",
]

def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured embedding provider."""
    cfg = settings.embedding
    if cfg.provider == "stub":
        return StubEmbeddingProvider(model=cfg.model, dimensions=cfg.dimensions)

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "dimensions": cfg.dimensions,
        "api_key": settings.require_api_key(cfg.api_key_env),
        "base_url": cfg.base_url,
        "batch_size": cfg.batch_size,
        "timeout": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
    }
    if cfg.provider == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    if cfg.provider == "voyage":
        return VoyageEmbeddingProvider(**kwargs)
    if cfg.provider == "cohere":
        return CohereEmbeddingProvider(**kwargs)
    raise ConfigError(f"unknown embedding provider: {cfg.provider!r}")
