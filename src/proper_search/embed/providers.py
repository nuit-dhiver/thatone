"""Concrete embedding providers.

Three vendors with three different request and response shapes. Each subclass
supplies only what differs; retries, batching, and error mapping come from
:class:`~proper_search.embed.base.HTTPEmbeddingProvider`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from ..errors import ProviderResponseInvalid
from .base import EmbeddingProvider, HTTPEmbeddingProvider, InputType


class OpenAIEmbeddingProvider(HTTPEmbeddingProvider):
    """OpenAI ``/v1/embeddings``.

    Symmetric: no query/document distinction, so ``input_type`` is accepted and
    ignored rather than silently mapped onto something that does not exist.
    """

    name = "openai"

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=base_url or "https://api.openai.com/v1", **kwargs)

    def _endpoint(self) -> str:
        return "/embeddings"

    def _payload(self, texts: Sequence[str], input_type: InputType) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        # Only the v3 models support truncation to a shorter width; sending it
        # to an older model is a 400.
        if "text-embedding-3" in self.model:
            payload["dimensions"] = self.dimensions
        return payload

    def _parse(self, data: dict[str, Any], expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderResponseInvalid(f"unexpected OpenAI response shape: {list(data)}")
        # The API documents that results may come back out of order, and the
        # caller zips these against chunk ids.
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


class VoyageEmbeddingProvider(HTTPEmbeddingProvider):
    """Voyage AI ``/v1/embeddings``. Asymmetric — honours ``input_type``."""

    name = "voyage"

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=base_url or "https://api.voyageai.com/v1", **kwargs)

    def _endpoint(self) -> str:
        return "/embeddings"

    def _payload(self, texts: Sequence[str], input_type: InputType) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": list(texts),
            "input_type": "query" if input_type is InputType.QUERY else "document",
            "output_dimension": self.dimensions,
        }

    def _parse(self, data: dict[str, Any], expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderResponseInvalid(f"unexpected Voyage response shape: {list(data)}")
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


class CohereEmbeddingProvider(HTTPEmbeddingProvider):
    """Cohere ``/v2/embed``. Asymmetric, and a different response shape again."""

    name = "cohere"

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=base_url or "https://api.cohere.com/v2", **kwargs)

    def _endpoint(self) -> str:
        return "/embed"

    def _payload(self, texts: Sequence[str], input_type: InputType) -> dict[str, Any]:
        return {
            "model": self.model,
            "texts": list(texts),
            "input_type": "search_query" if input_type is InputType.QUERY else "search_document",
            "embedding_types": ["float"],
        }

    def _parse(self, data: dict[str, Any], expected: int) -> list[list[float]]:
        embeddings = data.get("embeddings")
        if isinstance(embeddings, dict):  # v2 shape
            floats = embeddings.get("float")
            if isinstance(floats, list):
                return floats
        if isinstance(embeddings, list):  # v1 fallback
            return embeddings
        raise ProviderResponseInvalid(f"unexpected Cohere response shape: {list(data)}")


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings for tests and offline development.

    Not semantic — it cannot be — but it is stable, dependency-free, and
    genuinely discriminative: identical text embeds identically and shared
    words pull vectors together, which is enough to test fusion, ranking, and
    the dimension guards without a network or an API key.
    """

    name = "stub"

    def __init__(self, *, model: str = "stub-embed-1", dimensions: int = 64) -> None:
        self.model = model
        self.dimensions = dimensions

    def _vector(self, text: str, input_type: InputType) -> list[float]:
        vector = [0.0] * self.dimensions
        # A bag-of-words projection: each token contributes to a few fixed
        # buckets, so texts sharing vocabulary end up close together.
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            for offset in range(3):
                bucket = int.from_bytes(digest[offset * 4 : offset * 4 + 4], "big")
                sign = 1.0 if digest[offset] % 2 else -1.0
                vector[bucket % self.dimensions] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # An empty or purely-punctuation chunk still needs a valid unit
            # vector, or the KNN table rejects it.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    async def embed(
        self, texts: Sequence[str], *, input_type: InputType = InputType.DOCUMENT
    ) -> list[list[float]]:
        return [self._vector(text, input_type) for text in texts]
