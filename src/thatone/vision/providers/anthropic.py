"""Anthropic vision provider.

Uses the official SDK. Three constraints of the current models drive the shape
of this file, and getting any of them wrong fails the whole run rather than one
item:

* ``temperature`` / ``top_p`` / ``top_k`` are rejected outright. Output is
  steered by the prompt.
* ``budget_tokens`` is gone; depth is controlled by ``output_config.effort``.
* A declined request returns **HTTP 200** with ``stop_reason == "refusal"``, so
  reading ``response.content[0]`` without checking first raises IndexError on
  ordinary input. Over 100k user-supplied clips this will happen.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

from ...config import VisionSettings
from ...errors import (
    AuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderResponseInvalid,
    ProviderUnavailable,
)
from ...models import Description, UsageRecord
from ..base import VisionProvider, VisionRequest
from ..schema import to_api_schema

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

DESCRIPTION_SCHEMA = to_api_schema(Description)


class AnthropicVisionProvider(VisionProvider):
    """Vision descriptions via the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, settings: VisionSettings, api_key: str) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "the anthropic provider needs the anthropic SDK; "
                "install thatone[anthropic]"
            ) from exc

        self.settings = settings
        self.model = settings.model
        self._client: AsyncAnthropic = AsyncAnthropic(
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            # The SDK already retries 429/5xx/connection errors with backoff.
            # Leaving that on means the job queue only ever sees failures that
            # survived it, so its own backoff covers genuine outages rather
            # than duplicating transient-blip handling.
            max_retries=settings.max_retries,
        )

    # -- request construction ---------------------------------------------

    def _build_content(self, request: VisionRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for frame in request.frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": frame.media_type,
                        "data": base64.standard_b64encode(frame.image_bytes).decode("ascii"),
                    },
                }
            )
        # Text last: with the images already in context, the instruction reads
        # as being about them rather than as a preamble.
        content.append({"type": "text", "text": request.instruction})
        return content

    def _build_system(self, request: VisionRequest) -> list[dict[str, Any]] | None:
        if not request.system:
            return None
        block: dict[str, Any] = {"type": "text", "text": request.system}
        if request.cache_system and self.settings.cache_system_prompt:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _base_kwargs(self, request: VisionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.settings.max_tokens,
            "messages": [{"role": "user", "content": self._build_content(request)}],
        }
        system = self._build_system(request)
        if system:
            kwargs["system"] = system
        return kwargs

    # -- calls -------------------------------------------------------------

    async def describe(self, request: VisionRequest) -> tuple[Description, UsageRecord]:
        kwargs = self._base_kwargs(request)
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": DESCRIPTION_SCHEMA}
        }
        if self.settings.effort:
            output_config["effort"] = self.settings.effort
        kwargs["output_config"] = output_config

        response = await self._call(kwargs)
        text = self._extract_text(response)
        try:
            description = Description.model_validate_json(text)
        except ValueError as exc:
            raise ProviderResponseInvalid(
                f"{self.model} returned output that does not satisfy the description "
                f"contract: {exc}"
            ) from exc
        return description, self._usage(response)

    async def complete(self, request: VisionRequest) -> tuple[str, UsageRecord]:
        kwargs = self._base_kwargs(request)
        if self.settings.effort:
            kwargs["output_config"] = {"effort": self.settings.effort}
        response = await self._call(kwargs)
        return self._extract_text(response), self._usage(response)

    async def _call(self, kwargs: dict[str, Any]) -> Any:
        """Issue the request, mapping SDK errors onto the retry taxonomy."""
        import anthropic

        try:
            return await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            retry_after = None
            header = getattr(exc, "response", None)
            if header is not None:
                try:
                    retry_after = float(header.headers.get("retry-after", ""))
                except (TypeError, ValueError):
                    retry_after = None
            raise ProviderRateLimited(str(exc), retry_after=retry_after) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            # Not per-item: every subsequent call fails identically, so this is
            # terminal and the worker should stop rather than burn the queue.
            raise AuthError(f"{self.model}: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderBadRequest(f"{self.model} rejected the request: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(f"{self.model}: {exc}") from exc
            raise ProviderBadRequest(f"{self.model}: {exc}") from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise ProviderUnavailable(f"{self.model}: {exc}") from exc

    # -- response handling -------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text out, after checking why generation stopped.

        Order matters. ``stop_reason`` is checked before ``content`` is touched
        because a refusal returns a successful response whose content array is
        empty or partial.
        """
        stop_reason = getattr(response, "stop_reason", None)

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ProviderRefusal(
                f"the model declined to describe this media"
                f"{f' (category: {category})' if category else ''}",
                category=category,
            )

        if stop_reason == "max_tokens":
            # Structured output truncated mid-JSON. Retrying identically just
            # truncates again, so this is terminal with an actionable message.
            raise ProviderResponseInvalid(
                "the response hit max_tokens and the output is truncated; "
                "raise vision.max_tokens or lower sampling.max_frames"
            )

        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if not parts:
            raise ProviderResponseInvalid(
                f"response contained no text block (stop_reason={stop_reason!r})"
            )
        return "".join(parts)

    def _usage(self, response: Any) -> UsageRecord:
        usage = getattr(response, "usage", None)
        if usage is None:
            return UsageRecord(model=self.model)
        return UsageRecord(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=getattr(response, "model", self.model),
        )

    async def count_tokens(self, request: VisionRequest) -> int:
        """Real token count for a request, used by the cost estimator.

        The provider's own counter rather than an approximation: image tokens
        in particular do not follow a simple formula across resolutions, and a
        cost projection for a 100k-item run should not be built on a guess.
        """
        import anthropic

        kwargs = self._base_kwargs(request)
        kwargs.pop("max_tokens", None)
        try:
            result = await self._client.messages.count_tokens(**kwargs)
        except anthropic.APIStatusError as exc:
            raise ProviderUnavailable(f"token counting failed: {exc}") from exc
        return int(result.input_tokens)

    async def close(self) -> None:
        await self._client.close()


def schema_json() -> str:
    """The description schema as sent to the API. Handy when debugging."""
    return json.dumps(DESCRIPTION_SCHEMA, indent=2)
