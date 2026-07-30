"""Vision layer tests: schema adaptation, prompts, providers, strategies."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from thatone.config import VisionSettings
from thatone.errors import (
    ProviderRateLimited,
    ProviderRefusal,
    ProviderResponseInvalid,
)
from thatone.models import Description, FrameSample
from thatone.vision import prompts
from thatone.vision.base import VisionRequest, estimate_image_tokens
from thatone.vision.providers.stub import StubVisionProvider
from thatone.vision.schema import to_api_schema
from thatone.vision.strategies import (
    SequentialStrategy,
    SingleCallStrategy,
    TwoPassStrategy,
    get_strategy,
)


def frame(index: int = 0, t_ms: int = 0, payload: bytes = b"") -> FrameSample:
    return FrameSample(
        index=index,
        t_ms=t_ms,
        image_bytes=payload or bytes([index, index + 1, index + 2]) * 32,
        media_type="image/jpeg",
        width=480,
        height=270,
    )


# --------------------------------------------------------------------------
# Schema adaptation
# --------------------------------------------------------------------------


class TestSchemaAdapter:
    def test_every_field_is_required(self) -> None:
        """Optional fields invite omission, and an absent on_screen_text is
        indistinguishable from a clip that genuinely had none."""
        schema = to_api_schema(Description)
        assert set(schema["required"]) == set(schema["properties"])

    def test_objects_forbid_extra_properties(self) -> None:
        schema = to_api_schema(Description)
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["FrameNote"]["additionalProperties"] is False

    def test_unsupported_constraints_are_stripped(self) -> None:
        """Constraints the structured-output dialect rejects would fail the
        whole run, not one item, so they must never reach the API."""

        class Constrained(BaseModel):
            model_config = ConfigDict(extra="forbid")
            count: int = Field(ge=1, le=10)
            label: str = Field(min_length=2, max_length=8, pattern="^[a-z]+$")
            items: list[str] = Field(min_length=1, max_length=3)

        blob = json.dumps(to_api_schema(Constrained))
        for keyword in ("minimum", "maximum", "minLength", "maxLength", "pattern", "minItems"):
            assert keyword not in blob, f"{keyword} survived into the API schema"

    def test_class_docstrings_are_stripped_but_field_descriptions_survive(self) -> None:
        """Docstrings explain the implementation to maintainers; field
        descriptions are written for the model. Only the latter belong here."""
        schema = to_api_schema(Description)
        assert "description" not in schema
        assert "description" not in schema["$defs"]["FrameNote"]
        assert "verbatim" in schema["properties"]["on_screen_text"]["description"]

    def test_titles_and_defaults_are_dropped(self) -> None:
        blob = json.dumps(to_api_schema(Description))
        assert '"title"' not in blob
        assert '"default"' not in blob

    def test_enums_and_refs_survive(self) -> None:
        schema = to_api_schema(Description)
        assert schema["$defs"]["Confidence"]["enum"] == ["high", "medium", "low"]
        assert schema["properties"]["frame_notes"]["items"]["$ref"].endswith("FrameNote")

    def test_all_required_can_be_disabled(self) -> None:
        assert to_api_schema(Description, all_required=False)["required"] == ["narrative"]


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


class TestPrompts:
    def test_version_is_declared(self) -> None:
        assert prompts.PROMPT_VERSION

    def test_system_prompt_covers_every_contract_field(self) -> None:
        """The schema names the fields; the system prompt is what says how to
        fill them. A field described in one and not the other gets filled badly."""
        for field_name in Description.model_fields:
            assert field_name in prompts.SYSTEM, f"{field_name} is unexplained in the prompt"

    def test_single_call_instruction_lists_frame_timestamps(self) -> None:
        frames = [frame(0, 0), frame(1, 900), frame(2, 1800)]
        text = prompts.single_call_instruction(frames, duration_ms=2000)
        assert "3 frames" in text
        for f in frames:
            assert f"t={f.t_ms}ms" in text

    def test_single_frame_reads_naturally(self) -> None:
        text = prompts.single_call_instruction([frame(0, 0)], duration_ms=100)
        assert "1 frame" in text and "1 frames" not in text

    def test_sequential_continuation_licenses_rewriting(self) -> None:
        """Without explicit permission to overwrite, models append to the story
        they already told and an early misreading survives to the end."""
        text = prompts.sequential_next_instruction(
            frame(1, 900), position=2, total=4, running="A cat sits."
        )
        assert "A cat sits." in text
        assert "rewrite" in text.lower()
        assert "2 of 4" in text

    def test_caption_instruction_forbids_speculation(self) -> None:
        text = prompts.caption_instruction(frame(0, 500))
        assert "t=500ms" in text
        assert "speculate" in text.lower()

    def test_synthesis_instruction_carries_all_captions(self) -> None:
        text = prompts.synthesis_instruction([(0, "a cat"), (900, "a dog")], duration_ms=1800)
        assert "t=0ms: a cat" in text and "t=900ms: a dog" in text


# --------------------------------------------------------------------------
# Stub provider
# --------------------------------------------------------------------------


class TestStubProvider:
    async def test_is_deterministic(self) -> None:
        """Search assertions depend on identical input yielding identical text."""
        provider = StubVisionProvider()
        request = VisionRequest(instruction="go", frames=[frame(0), frame(1)])
        first, _ = await provider.describe(request)
        second, _ = await provider.describe(request)
        assert first.narrative == second.narrative
        assert first.tags == second.tags

    async def test_different_frames_produce_different_descriptions(self) -> None:
        provider = StubVisionProvider()
        a, _ = await provider.describe(VisionRequest(instruction="x", frames=[frame(0)]))
        b, _ = await provider.describe(
            VisionRequest(instruction="x", frames=[frame(9, payload=b"totally different" * 8)])
        )
        assert a.narrative != b.narrative

    async def test_emits_one_note_per_frame(self) -> None:
        provider = StubVisionProvider()
        frames = [frame(i, i * 100) for i in range(4)]
        description, _ = await provider.describe(VisionRequest(instruction="x", frames=frames))
        assert len(description.frame_notes) == 4
        assert [n.t_ms for n in description.frame_notes] == [0, 100, 200, 300]

    async def test_reports_usage(self) -> None:
        provider = StubVisionProvider()
        _, usage = await provider.describe(
            VisionRequest(instruction="x" * 100, frames=[frame(0), frame(1)])
        )
        assert usage.input_tokens > 0 and usage.output_tokens > 0

    async def test_records_calls(self) -> None:
        provider = StubVisionProvider()
        await provider.describe(VisionRequest(instruction="first", frames=[frame(0)]))
        await provider.describe(VisionRequest(instruction="second", frames=[frame(1)]))
        assert [c.instruction for c in provider.calls] == ["first", "second"]

    async def test_permanent_failure_injection(self) -> None:
        provider = StubVisionProvider(fail_with=ProviderRefusal("nope"))
        with pytest.raises(ProviderRefusal):
            await provider.describe(VisionRequest(instruction="x", frames=[frame(0)]))

    async def test_transient_failure_injection_recovers(self) -> None:
        """Exercises the retry path without waiting on a real outage."""
        provider = StubVisionProvider(fail_with=ProviderRateLimited("slow down"), fail_times=2)
        request = VisionRequest(instruction="x", frames=[frame(0)])
        for _ in range(2):
            with pytest.raises(ProviderRateLimited):
                await provider.describe(request)
        description, _ = await provider.describe(request)
        assert description.narrative


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


class TestStrategies:
    frames: ClassVar[list[FrameSample]] = [frame(i, i * 500) for i in range(4)]

    async def test_single_call_makes_exactly_one_request(self) -> None:
        """One request per item is the whole cost argument for this default."""
        provider = StubVisionProvider()
        result = await SingleCallStrategy().run(provider, self.frames, duration_ms=2000)
        assert len(provider.calls) == 1
        assert len(provider.calls[0].frames) == 4, "all frames must go in one message"
        assert result.strategy == "single_call"
        assert result.description.narrative

    async def test_sequential_makes_one_request_per_frame(self) -> None:
        provider = StubVisionProvider()
        result = await SequentialStrategy().run(provider, self.frames, duration_ms=2000)
        assert len(provider.calls) == 4
        assert all(len(c.frames) == 1 for c in provider.calls)
        assert result.strategy == "sequential"

    async def test_sequential_feeds_the_running_description_forward(self) -> None:
        """The revision behaviour depends on each turn seeing the prior story."""
        provider = StubVisionProvider()
        await SequentialStrategy().run(provider, self.frames, duration_ms=2000)
        assert "Narrative:" not in provider.calls[0].instruction, "first turn has no history"
        for call in provider.calls[1:]:
            assert "Narrative:" in call.instruction
            assert "Frame notes:" in call.instruction

    async def test_sequential_accumulates_usage_across_turns(self) -> None:
        provider = StubVisionProvider()
        single = await SingleCallStrategy().run(provider, self.frames, duration_ms=2000)
        provider.calls.clear()
        multi = await SequentialStrategy().run(provider, self.frames, duration_ms=2000)
        assert multi.usage.input_tokens > single.usage.input_tokens, (
            "per-frame turns must report their true cumulative cost"
        )

    async def test_two_pass_captions_every_frame_then_synthesizes(self) -> None:
        provider = StubVisionProvider()
        result = await TwoPassStrategy().run(provider, self.frames, duration_ms=2000)
        assert len(provider.calls) == 5, "expected 4 captions + 1 synthesis"
        assert result.strategy == "two_pass"

    async def test_two_pass_synthesis_does_not_resend_images(self) -> None:
        """Re-sending frames would double image-token cost for the one pass
        whose entire purpose is that it reasons over text instead."""
        provider = StubVisionProvider()
        await TwoPassStrategy().run(provider, self.frames, duration_ms=2000)
        assert len(provider.calls[-1].frames) == 0

    async def test_two_pass_can_route_captions_to_a_cheaper_model(self) -> None:
        strong = StubVisionProvider(model="strong")
        cheap = StubVisionProvider(model="cheap")
        await TwoPassStrategy().run(
            strong, self.frames, duration_ms=2000, caption_provider=cheap
        )
        assert len(cheap.calls) == 4, "captions should go to the cheap model"
        assert len(strong.calls) == 1, "only the synthesis needs the strong model"

    @pytest.mark.parametrize("name", ["single_call", "sequential", "two_pass"])
    async def test_every_strategy_produces_a_usable_description(self, name: str) -> None:
        result = await get_strategy(name).run(
            StubVisionProvider(), self.frames, duration_ms=2000
        )
        assert result.description.narrative
        assert result.description.tags
        assert result.model and result.strategy == name

    @pytest.mark.parametrize("name", ["single_call", "sequential", "two_pass"])
    async def test_no_frames_is_rejected(self, name: str) -> None:
        with pytest.raises(ProviderResponseInvalid, match="no sampled frames"):
            await get_strategy(name).run(StubVisionProvider(), [], duration_ms=0)

    def test_unknown_strategy_lists_the_options(self) -> None:
        with pytest.raises(ValueError, match="single_call"):
            get_strategy("telepathy")


# --------------------------------------------------------------------------
# Anthropic response handling
# --------------------------------------------------------------------------


def fake_response(
    *, stop_reason: str = "end_turn", text: str | None = None, category: str | None = None
) -> Any:
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=content,
        model="claude-sonnet-5",
        usage=SimpleNamespace(
            input_tokens=1700,
            output_tokens=400,
            cache_read_input_tokens=1200,
            cache_creation_input_tokens=0,
        ),
    )


@pytest.fixture
def anthropic_provider():
    pytest.importorskip("anthropic")
    from thatone.vision.providers.anthropic import AnthropicVisionProvider

    # Never issues a request; only the response-handling logic is exercised.
    return AnthropicVisionProvider(VisionSettings(model="claude-sonnet-5"), api_key="test-key")


class TestAnthropicResponseHandling:
    def test_refusal_raises_before_content_is_touched(self, anthropic_provider: Any) -> None:
        """A decline is HTTP 200 with an empty content array. Reading
        content[0] first would raise IndexError instead of a typed refusal,
        and over 100k user-supplied clips this path will be hit."""
        with pytest.raises(ProviderRefusal) as caught:
            anthropic_provider._extract_text(fake_response(stop_reason="refusal", category="cyber"))
        assert caught.value.category == "cyber"

    def test_refusal_is_terminal_not_retryable(self, anthropic_provider: Any) -> None:
        """Retrying sends identical frames to an identical classifier."""
        from thatone.errors import RetryableError, TerminalError

        assert issubclass(ProviderRefusal, TerminalError)
        assert not issubclass(ProviderRefusal, RetryableError)

    def test_truncated_output_is_terminal_with_an_actionable_message(
        self, anthropic_provider: Any
    ) -> None:
        with pytest.raises(ProviderResponseInvalid, match="max_tokens"):
            anthropic_provider._extract_text(fake_response(stop_reason="max_tokens", text='{"nar'))

    def test_empty_content_is_reported_clearly(self, anthropic_provider: Any) -> None:
        with pytest.raises(ProviderResponseInvalid, match="no text block"):
            anthropic_provider._extract_text(fake_response())

    def test_text_is_extracted(self, anthropic_provider: Any) -> None:
        assert anthropic_provider._extract_text(fake_response(text='{"a":1}')) == '{"a":1}'

    def test_usage_maps_cache_fields(self, anthropic_provider: Any) -> None:
        """cache_read_input_tokens is the only proof prompt caching engaged."""
        usage = anthropic_provider._usage(fake_response(text="x"))
        assert usage.input_tokens == 1700
        assert usage.output_tokens == 400
        assert usage.cache_read_tokens == 1200
        assert usage.model == "claude-sonnet-5"

    def test_request_sends_images_before_the_instruction(self, anthropic_provider: Any) -> None:
        """With the images already in context, the instruction reads as being
        about them rather than as a preamble."""
        content = anthropic_provider._build_content(
            VisionRequest(instruction="describe", frames=[frame(0), frame(1)])
        )
        assert [c["type"] for c in content] == ["image", "image", "text"]
        assert content[-1]["text"] == "describe"
        assert content[0]["source"]["type"] == "base64"

    def test_system_prompt_is_marked_cacheable(self, anthropic_provider: Any) -> None:
        blocks = anthropic_provider._build_system(VisionRequest(instruction="x", system="SYS"))
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_caching_can_be_disabled_per_request(self, anthropic_provider: Any) -> None:
        blocks = anthropic_provider._build_system(
            VisionRequest(instruction="x", system="SYS", cache_system=False)
        )
        assert "cache_control" not in blocks[0]

    def test_no_sampling_parameters_are_ever_sent(self, anthropic_provider: Any) -> None:
        """temperature/top_p/top_k are rejected outright by these models."""
        kwargs = anthropic_provider._base_kwargs(
            VisionRequest(instruction="x", system="SYS", frames=[frame(0)])
        )
        assert not ({"temperature", "top_p", "top_k"} & set(kwargs))

    def test_no_budget_tokens_are_ever_sent(self, anthropic_provider: Any) -> None:
        kwargs = anthropic_provider._base_kwargs(VisionRequest(instruction="x"))
        assert "thinking" not in kwargs and "budget_tokens" not in kwargs


class TestTokenEstimate:
    def test_scales_with_frame_area(self) -> None:
        small = [FrameSample(index=0, t_ms=0, image_bytes=b"", width=240, height=135)]
        large = [FrameSample(index=0, t_ms=0, image_bytes=b"", width=960, height=540)]
        assert estimate_image_tokens(large) > 10 * estimate_image_tokens(small)

    def test_scales_with_frame_count(self) -> None:
        one = [FrameSample(index=0, t_ms=0, image_bytes=b"", width=480, height=270)]
        assert estimate_image_tokens(one * 6) == 6 * estimate_image_tokens(one)
