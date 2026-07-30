"""Fetching remote media.

URLs here are attacker-controlled in any realistic deployment: this is a tool
others run, and "index these links" is the normal way to use it. That makes the
fetcher a server-side request forgery surface, so it is written defensively.

Four protections, each covering a distinct failure:

* **Scheme allowlist** — ``file://``, ``gopher://``, and friends never reach a
  handler.
* **Address filtering** — the resolved IP must be public. This blocks
  ``169.254.169.254`` (cloud instance metadata, the classic SSRF payoff),
  loopback, and RFC1918 ranges.
* **Per-hop re-validation** — redirects are followed manually, because a
  public URL that 302s to ``127.0.0.1`` defeats a check performed only on the
  original URL.
* **Streaming size cap** — the download aborts mid-transfer once the limit is
  passed, so a lying (or absent) ``Content-Length`` cannot fill the disk.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..config import FetchSettings
from ..errors import MediaError, MediaTooLargeError, RetryableError, TerminalError
from ..media.hashing import content_hash


class UnsafeURLError(TerminalError):
    """The URL points somewhere a server-side fetcher must not go."""


@dataclass(slots=True)
class FetchResult:
    path: Path
    content_hash: str
    size_bytes: int
    content_type: str
    final_url: str


def _is_public_address(host: str) -> bool:
    """Resolve a hostname and require every address to be public.

    Every address, not just the first: a hostname resolving to both a public
    and a private address would otherwise pass the check and then connect to
    whichever the OS picks.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local  # 169.254.0.0/16 — cloud metadata lives here
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


def validate_url(url: str, settings: FetchSettings) -> str:
    """Check scheme and destination. Returns the host, or raises."""
    parsed = urlparse(url)
    if parsed.scheme not in settings.allowed_schemes:
        raise UnsafeURLError(
            f"scheme {parsed.scheme!r} is not allowed; permitted: "
            f"{', '.join(settings.allowed_schemes)}"
        )
    if not parsed.hostname:
        raise UnsafeURLError(f"no host in URL: {url}")
    if not _is_public_address(parsed.hostname):
        raise UnsafeURLError(
            f"{parsed.hostname} resolves to a non-public address; refusing to fetch. "
            f"Private, loopback, and link-local destinations are blocked because a "
            f"server-side fetcher reaching them is a request-forgery vector."
        )
    return parsed.hostname


class MediaFetcher:
    """Downloads remote media into the blob cache."""

    def __init__(self, settings: FetchSettings, blob_dir: str | Path) -> None:
        self.settings = settings
        self.blob_dir = Path(blob_dir).expanduser()
        self._client = httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            headers={"User-Agent": settings.user_agent},
            # Redirects are followed by hand so each hop can be re-validated.
            follow_redirects=False,
        )

    async def fetch(self, url: str) -> FetchResult:
        """Download a URL to the blob cache, validating every redirect hop."""
        current = url
        for _ in range(self.settings.max_redirects + 1):
            validate_url(current, self.settings)
            try:
                async with self._client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise MediaError(f"{current} returned a redirect with no location")
                        current = str(httpx.URL(current).join(location))
                        continue

                    if response.status_code == 429 or response.status_code >= 500:
                        retry_after = response.headers.get("retry-after")
                        raise RetryableError(
                            f"{current} returned {response.status_code}",
                            retry_after=float(retry_after) if retry_after else None,
                        )
                    if response.status_code >= 400:
                        raise MediaError(f"{current} returned {response.status_code}")

                    declared = response.headers.get("content-length")
                    if declared and int(declared) > self.settings.max_bytes:
                        raise MediaTooLargeError(
                            f"{current} declares {declared} bytes, over the "
                            f"{self.settings.max_bytes} limit"
                        )

                    payload = await self._read_capped(response, current)
                    content_type = (
                        response.headers.get("content-type", "").split(";")[0].strip()
                    )
                    return self._store(payload, content_type, str(response.url))
            except httpx.TimeoutException as exc:
                raise RetryableError(f"{current} timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise RetryableError(f"{current} failed: {exc}") from exc

        raise MediaError(f"too many redirects starting from {url}")

    async def _read_capped(self, response: httpx.Response, url: str) -> bytes:
        """Read the body, aborting as soon as the cap is exceeded.

        Checked while streaming rather than after: trusting Content-Length
        means a server that lies about it (or omits it) can write an unbounded
        file to disk.
        """
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.settings.max_bytes:
                raise MediaTooLargeError(
                    f"{url} exceeded the {self.settings.max_bytes} byte limit mid-download"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _store(self, payload: bytes, content_type: str, final_url: str) -> FetchResult:
        if not payload:
            raise MediaError(f"{final_url} returned an empty body")
        digest = content_hash(payload)
        # Sharded by hash prefix, matching the thumbnail layout, so no single
        # directory accumulates 100k entries.
        directory = self.blob_dir / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / digest
        if not path.exists():
            # Write to a temporary name and rename: a crash mid-write would
            # otherwise leave a truncated blob at the canonical path, which
            # later reads would trust because the name implies the content.
            staging = path.with_suffix(".partial")
            staging.write_bytes(payload)
            staging.replace(path)
        return FetchResult(
            path=path,
            content_hash=digest,
            size_bytes=len(payload),
            content_type=content_type,
            final_url=final_url,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> MediaFetcher:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
