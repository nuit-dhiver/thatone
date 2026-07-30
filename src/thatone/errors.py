"""Exception hierarchy.

The split that matters operationally is :class:`RetryableError` vs
:class:`TerminalError`. The job worker branches on exactly that: retryable
errors go back on the queue with backoff, terminal ones are recorded against
the item and never retried. Everything else is treated as a bug and allowed to
propagate.
"""

from __future__ import annotations


class ThatOneError(Exception):
    """Base class for every error this package raises deliberately."""


# --------------------------------------------------------------------------
# Retry classification
# --------------------------------------------------------------------------


class RetryableError(ThatOneError):
    """A transient failure. The worker should back off and try again.

    ``retry_after`` carries a provider-supplied hint in seconds when one is
    available (e.g. an HTTP ``retry-after`` header); the worker prefers it over
    its own backoff schedule.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TerminalError(ThatOneError):
    """A permanent failure for this item. Retrying would fail identically."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class ConfigError(ThatOneError):
    """Configuration is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------


class MediaError(TerminalError):
    """The media itself is the problem — it will not decode on a retry."""


class DecodeError(MediaError):
    """The file could not be decoded into frames."""


class UnsupportedMediaError(MediaError):
    """The file decoded, but is not a media type this pipeline handles."""


class MediaTooLargeError(MediaError):
    """The file exceeds the configured size or duration ceiling."""


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class ProviderError(ThatOneError):
    """Base for vision and embedding provider failures."""


class ProviderRateLimited(RetryableError, ProviderError):
    """Provider returned 429 or an explicit rate-limit signal."""


class ProviderUnavailable(RetryableError, ProviderError):
    """Provider returned 5xx, timed out, or the connection failed."""


class ProviderRefusal(TerminalError, ProviderError):
    """The model declined to describe this media.

    This is a successful HTTP response, not an error status: Anthropic returns
    200 with ``stop_reason == "refusal"``. Retrying sends the identical frames
    into the identical classifier, so it is terminal by definition. Over a
    100k-item corpus a handful of these is expected and must not stop the run.
    """

    def __init__(self, message: str, *, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class ProviderBadRequest(TerminalError, ProviderError):
    """The request was malformed or rejected — a bug in our request, not a blip."""


class ProviderResponseInvalid(TerminalError, ProviderError):
    """The provider replied, but the payload did not satisfy the output contract."""


class AuthError(TerminalError, ProviderError):
    """Missing or rejected credentials. Fails the whole run, not just one item."""


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


class StorageError(ThatOneError):
    """Base for storage-backend failures."""


class SchemaVersionError(StorageError):
    """The database was written by a different schema version than this code."""


class IndexConsistencyError(StorageError):
    """A search would compare vectors that are not comparable.

    Raised when the embedding model or dimension recorded at index time does
    not match the one configured now. Silently proceeding here returns results
    that look plausible and are meaningless, so this always raises.
    """


class ExtensionUnavailableError(StorageError):
    """The SQLite build cannot load the ``sqlite-vec`` extension.

    Some Python distributions are compiled without ``enable_load_extension``.
    Vector search is impossible on those builds; the message points at the fix.
    """
