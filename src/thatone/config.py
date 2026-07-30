"""Configuration.

Layered, highest priority first: explicit keyword arguments, environment
variables, a YAML file, then defaults. Nested values use a double underscore,
so ``THATONE__VISION__MODEL=claude-opus-5`` sets ``vision.model``.

Two rules this module exists to enforce:

* **No secrets in config values.** Credentials are named by environment
  variable (``api_key_env``), never inlined, so a config file is safe to commit.
* **No hardcoded prices.** Provider rates change; they live in
  :class:`PricingSettings` so a cost estimate can be corrected without a
  release.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .errors import ConfigError

ENV_PREFIX = "THATONE__"
DEFAULT_CONFIG_FILENAMES = ("thatone.yaml", "thatone.yml", ".thatone.yaml")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: Path = Path("./thatone.db")
    """SQLite database file. Ignored when ``backend`` is postgres."""

    dsn_env: str | None = None
    """Environment variable holding the Postgres DSN. Required for postgres."""

    blob_dir: Path = Path("./thatone-blobs")
    """Where fetched media and generated thumbnails are cached."""

    busy_timeout_ms: int = 30_000
    """How long a writer waits on a locked SQLite database before erroring.

    Concurrent ingest workers contend on write; the default of 0 turns routine
    contention into spurious ``database is locked`` failures.
    """


class SamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["adaptive", "interval", "count"] = "adaptive"

    min_frames: int = Field(default=3, ge=1)
    max_frames: int = Field(default=12, ge=1)
    """Ceiling on frames per item. This is the main cost dial: request cost is
    roughly linear in frame count."""

    hamming_threshold: int = Field(default=12, ge=0, le=64)
    """dHash distance at which a frame counts as a new scene. Lower keeps more
    frames. 12/64 is a reasonable 'visibly different' bar."""

    interval_seconds: float = Field(default=2.0, gt=0)
    target_count: int = Field(default=8, ge=1)

    max_decode_frames: int = Field(default=3000, ge=1)
    """Hard stop on frames decoded per item, so a pathological file cannot
    exhaust memory during sampling."""

    frame_max_edge: int = Field(default=768, ge=64)
    """Longest edge, in pixels, of frames sent to the model. Image tokens scale
    with area (~w*h/750), so this is the other cost dial."""

    jpeg_quality: int = Field(default=85, ge=1, le=100)

    @model_validator(mode="after")
    def _check_frame_bounds(self) -> SamplingSettings:
        if self.min_frames > self.max_frames:
            raise ValueError(
                f"sampling.min_frames ({self.min_frames}) exceeds "
                f"sampling.max_frames ({self.max_frames})"
            )
        return self


class VisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "gemini", "openai_compat", "stub"] = "anthropic"
    model: str = "claude-sonnet-5"
    strategy: Literal["single_call", "sequential", "two_pass"] = "single_call"

    base_url: str | None = None
    """Override the provider endpoint. This is how a self-hosted model is used:
    point ``openai_compat`` at your own server. This package never runs a model
    itself."""

    api_key_env: str = "ANTHROPIC_API_KEY"
    """Name of the environment variable holding the key — never the key."""

    max_tokens: int = Field(default=4096, ge=256)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    """Anthropic effort level. Frame description is extraction, not deep
    reasoning, so the default sits low; raise it if narratives read shallow."""

    timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    cache_system_prompt: bool = True
    """Cache the (long, identical) extraction prompt across calls. Only pays off
    above the model's minimum cacheable prefix — see ``prompt_cache_min_tokens``."""

    use_batch_api: bool = False
    """Route bulk description through the Batch API: half price, results within
    the hour instead of seconds. The right default for a backfill, wrong for
    interactive indexing, so it is opt-in per run."""

    fallback_model: str | None = None
    """Model to retry on when the primary declines a request."""

    # Two-pass strategy only.
    caption_model: str | None = None
    """Cheap model for the per-frame pass. Falls back to ``model`` when unset."""


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "voyage", "cohere", "stub"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = Field(default=1536, ge=8)
    """Must match the model's real output width. Recorded in the database at
    index time and checked on every search."""

    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = Field(default=128, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    max_chunk_chars: int = Field(default=1200, ge=100)
    """Chunks longer than this are split. Frame notes are naturally short; long
    narratives get segmented so one vector covers one idea."""


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_limit: int = Field(default=20, ge=1)
    candidate_limit: int = Field(default=200, ge=1)
    """How deep each signal retrieves before fusion. Larger costs little and
    materially improves recall on vague queries."""

    rrf_k: int = Field(default=60, ge=1)
    """Reciprocal-rank-fusion constant. 60 is the value from the original TREC
    work and is a sane default; lower sharpens toward top-ranked items."""

    enable_lexical: bool = True
    enable_dense: bool = True

    rerank: Literal["none", "llm"] = "none"
    rerank_model: str = "claude-haiku-4-5"
    rerank_top_n: int = Field(default=40, ge=1)

    # BM25 column weights. Lexical relevance for these columns is not equal: a
    # remembered verbatim caption is a far stronger signal than an incidental
    # word in a narrative, so on-screen text outweighs everything else.
    weight_narrative: float = 1.0
    weight_on_screen_text: float = 3.0
    weight_tags: float = 1.5

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> SearchSettings:
        if not self.enable_lexical and not self.enable_dense:
            raise ValueError(
                "search.enable_lexical and search.enable_dense are both false; "
                "at least one retrieval signal must be on"
            )
        return self


class FetchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_duration_ms: int = Field(default=10 * 60 * 1000, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_redirects: int = Field(default=5, ge=0)
    user_agent: str = "thatone/0.1"
    allowed_schemes: tuple[str, ...] = ("http", "https")
    concurrency: int = Field(default=8, ge=1)


class JobSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: Literal["embedded", "celery"] = "embedded"
    concurrency: int = Field(default=4, ge=1)
    """Simultaneous in-flight provider calls. Tune against your rate limit."""

    lease_seconds: int = Field(default=900, ge=1)
    """How long a claimed job stays claimed. Must exceed the slowest realistic
    call, or a healthy worker's job gets stolen mid-flight and duplicated."""

    max_attempts: int = Field(default=4, ge=1)
    backoff_base_seconds: float = Field(default=2.0, gt=0)
    backoff_max_seconds: float = Field(default=300.0, gt=0)
    claim_batch_size: int = Field(default=32, ge=1)


class ModelPrice(BaseModel):
    """Per-million-token rates. Kept in config because published rates move."""

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch_discount: float = 1.0,
    ) -> float:
        """Dollar cost for one call's token usage."""
        read_rate = (
            self.cache_read_per_mtok
            if self.cache_read_per_mtok is not None
            else self.input_per_mtok * 0.1
        )
        write_rate = (
            self.cache_write_per_mtok
            if self.cache_write_per_mtok is not None
            else self.input_per_mtok * 1.25
        )
        total = (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * read_rate
            + cache_write_tokens * write_rate
        ) / 1_000_000
        return total * batch_discount


class PricingSettings(BaseModel):
    """Rates used by the cost estimator.

    Defaults reflect published list prices at the time of writing and are
    expected to drift. Override in YAML rather than editing this file.
    """

    model_config = ConfigDict(extra="forbid")

    batch_discount: float = Field(default=0.5, gt=0, le=1)

    models: dict[str, ModelPrice] = Field(
        default_factory=lambda: {
            "claude-opus-5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
            "claude-sonnet-5": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
            "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
        }
    )

    embedding_per_mtok: float = 0.02
    """Embedding rate. Varies widely by provider; set it to yours."""

    def for_model(self, model: str) -> ModelPrice | None:
        return self.models.get(model)


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------


class Settings(BaseSettings):
    """Everything, assembled."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    storage: StorageSettings = Field(default_factory=StorageSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Explicit args beat env, env beats the YAML file, YAML beats defaults.
        return (init_settings, env_settings, _YamlSource(settings_cls))

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Settings:
        """Build settings, optionally from an explicit YAML path."""
        if path is not None:
            resolved = Path(path).expanduser()
            if not resolved.is_file():
                raise ConfigError(f"config file not found: {resolved}")
            os.environ[_YamlSource.PATH_ENV] = str(resolved)
        try:
            return cls(**overrides)
        finally:
            if path is not None:
                os.environ.pop(_YamlSource.PATH_ENV, None)

    def require_api_key(self, env_var: str) -> str:
        """Read a credential from the environment, or fail with a usable message."""
        value = os.environ.get(env_var)
        if not value:
            raise ConfigError(
                f"environment variable {env_var} is not set. "
                f"Export it, or point the relevant `api_key_env` at a different variable."
            )
        return value

    def describe(self) -> dict[str, Any]:
        """Config summary safe to log or return over HTTP.

        Only variable *names* are ever present in config, so there is nothing to
        redact — but this keeps the surface deliberate rather than incidental.
        """
        return {
            "storage": {"backend": self.storage.backend, "path": str(self.storage.path)},
            "vision": {
                "provider": self.vision.provider,
                "model": self.vision.model,
                "strategy": self.vision.strategy,
                "batch": self.vision.use_batch_api,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
            },
            "sampling": {
                "strategy": self.sampling.strategy,
                "min_frames": self.sampling.min_frames,
                "max_frames": self.sampling.max_frames,
            },
            "search": {
                "rrf_k": self.search.rrf_k,
                "rerank": self.search.rerank,
            },
        }


class _YamlSource(PydanticBaseSettingsSource):
    """Loads settings from a YAML file.

    Path resolution: ``THATONE_CONFIG`` if set, otherwise the first known
    filename found in the working directory. Absent file means empty mapping —
    running with no config at all is a supported path.
    """

    PATH_ENV = "THATONE_CONFIG"

    def _resolve_path(self) -> Path | None:
        explicit = os.environ.get(self.PATH_ENV)
        if explicit:
            return Path(explicit).expanduser()
        for name in DEFAULT_CONFIG_FILENAMES:
            candidate = Path.cwd() / name
            if candidate.is_file():
                return candidate
        return None

    def __call__(self) -> dict[str, Any]:
        path = self._resolve_path()
        if path is None:
            return {}
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {path}: {exc}") from exc
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{path} must contain a mapping at the top level, got {type(raw).__name__}"
            )
        return raw

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Not used: __call__ returns the whole mapping at once.
        raise NotImplementedError


__all__ = [
    "EmbeddingSettings",
    "FetchSettings",
    "JobSettings",
    "ModelPrice",
    "PricingSettings",
    "SamplingSettings",
    "SearchSettings",
    "Settings",
    "StorageSettings",
    "VisionSettings",
]
