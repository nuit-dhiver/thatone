"""Embedding provider interface.

All providers are HTTP clients built on ``httpx`` rather than three vendor
SDKs: the embedding endpoints are simple enough that one shared retry and
error-mapping path is worth more than per-vendor conveniences, and it keeps the
dependency footprint of a library small.

The one non-obvious part of the interface is :class:`InputType`. Several
providers embed a *query* and a *document* into deliberately different regions
of the space, and using the wrong one silently costs recall — no error, just
worse results — so it is a required part of the call rather than an option.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import httpx

from ..errors import (
    AuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderResponseInvalid,
    ProviderUnavailable,
)


class InputType(StrEnum):
    """Whether text is being stored or searched with.

    Asymmetric-embedding models place a question and its answer near each other
    rather than placing similar *texts* near each other. Labelling the side is
    what makes that work.
    """

    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingProvider(ABC):
    """Turns text into vectors."""

    name: str
    model: str
    dimensions: int

    @abstractmethod
    async def embed(
        self, texts: Sequence[str], *, input_type: InputType = InputType.DOCUMENT
    ) -> list[list[float]]:
        """Embed a batch, returning one vector per input in the same order.

        Order is load-bearing: callers zip the result back against chunk ids,
        so a provider that reorders its response must sort it before returning.
        """

    async def embed_one(
        self, text: str, *, input_type: InputType = InputType.QUERY
    ) -> list[float]:
        """Embed a single text. Defaults to query semantics — the common case
        for a lone embedding is a search box."""
        vectors = await self.embed([text], input_type=input_type)
        return vectors[0]

    async def close(self) -> None:  # noqa: B027
        """Release the HTTP client. Empty by default for providers without one."""

    async def __aenter__(self) -> EmbeddingProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class HTTPEmbeddingProvider(EmbeddingProvider):
    """Shared HTTP plumbing: batching, retries, and error mapping.

    Subclasses supply the request body and response shape; everything about
    *when to retry* lives here so the three providers cannot drift apart on the
    part that matters operationally.
    """

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str,
        base_url: str,
        batch_size: int = 128,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=self._auth_headers(api_key),
        )

    def _auth_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    @abstractmethod
    def _endpoint(self) -> str: ...

    @abstractmethod
    def _payload(self, texts: Sequence[str], input_type: InputType) -> dict[str, Any]: ...

    @abstractmethod
    def _parse(self, data: dict[str, Any], expected: int) -> list[list[float]]: ...

    async def embed(
        self, texts: Sequence[str], *, input_type: InputType = InputType.DOCUMENT
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(await self._embed_batch(batch, input_type))
        return vectors

    async def _embed_batch(
        self, texts: Sequence[str], input_type: InputType
    ) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    self._endpoint(), json=self._payload(texts, input_type)
                )
                self._raise_for_status(response)
                vectors = self._parse(response.json(), len(texts))
            except (ProviderRateLimited, ProviderUnavailable) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt, exc))
                continue
            except httpx.TimeoutException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ProviderUnavailable(f"{self.name} timed out: {exc}") from exc
                await asyncio.sleep(self._backoff(attempt, None))
                continue
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"{self.name} connection failed: {exc}") from exc

            if len(vectors) != len(texts):
                raise ProviderResponseInvalid(
                    f"{self.name} returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            for vector in vectors:
                if len(vector) != self.dimensions:
                    raise ProviderResponseInvalid(
                        f"{self.name} returned {len(vector)}-dimension vectors but "
                        f"embedding.dimensions is set to {self.dimensions}; correct the "
                        f"config, because the index records this width and will refuse "
                        f"to mix"
                    )
            return vectors

    @staticmethod
    def _backoff(attempt: int, exc: Exception | None) -> float:
        """Exponential backoff, honouring a server hint when there is one.

        Jitter matters here: a batch run fans out many concurrent embed calls,
        and without it they retry in lockstep and re-trigger the same limit.
        """
        hinted = getattr(exc, "retry_after", None)
        if hinted:
            return float(hinted)
        return min(30.0, 2.0**attempt) * (0.5 + random.random())

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = response.text[:400]
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimited(
                f"{self.name} rate limited: {body}",
                retry_after=float(retry_after) if retry_after else None,
            )
        if response.status_code in (401, 403):
            raise AuthError(f"{self.name} rejected the credentials: {body}")
        if response.status_code >= 500:
            raise ProviderUnavailable(f"{self.name} returned {response.status_code}: {body}")
        raise ProviderBadRequest(f"{self.name} returned {response.status_code}: {body}")

    async def close(self) -> None:
        await self._client.aclose()
