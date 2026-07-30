"""Where media comes from.

A source yields :class:`~thatone.models.MediaRef` values — candidates for
indexing, before any bytes have been read. Keeping discovery separate from
fetching means a directory walk over 100k files does not hold 100k file handles
or buffers, and a URL list can be validated before anything is downloaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from ..models import MediaRef, SourceType

MEDIA_SUFFIXES = frozenset(
    {".gif", ".webp", ".apng", ".png", ".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi"}
)


class MediaSourceProvider(ABC):
    """Discovers candidates for indexing."""

    @abstractmethod
    def __iter__(self) -> Iterator[MediaRef]: ...


def iter_local_media(
    root: str | Path,
    *,
    recursive: bool = True,
    suffixes: frozenset[str] = MEDIA_SUFFIXES,
) -> Iterator[MediaRef]:
    """Walk a directory, yielding media files.

    Lazy by design: a generator means a 100k-file tree starts producing work
    immediately instead of after a full walk, so ingest and discovery overlap.
    """
    base = Path(root).expanduser()
    if base.is_file():
        yield MediaRef(
            source_type=SourceType.LOCAL, source_uri=str(base.resolve()), local_path=str(base)
        )
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(base.glob(pattern)):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield MediaRef(
                source_type=SourceType.LOCAL,
                source_uri=str(path.resolve()),
                local_path=str(path),
            )


class LocalDirectorySource(MediaSourceProvider):
    """Media files found under a directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
        suffixes: frozenset[str] = MEDIA_SUFFIXES,
    ) -> None:
        self.root = Path(root).expanduser()
        self.recursive = recursive
        self.suffixes = suffixes

    def __iter__(self) -> Iterator[MediaRef]:
        return iter_local_media(self.root, recursive=self.recursive, suffixes=self.suffixes)


class UrlSource(MediaSourceProvider):
    """An explicit list of URLs.

    Nothing is validated or fetched here — that happens in
    :class:`~thatone.ingest.fetch.MediaFetcher`, which re-checks every
    redirect hop rather than trusting a URL that passed inspection once.
    """

    def __init__(self, urls: Sequence[str] | Iterable[str]) -> None:
        self.urls = list(urls)

    def __iter__(self) -> Iterator[MediaRef]:
        for url in self.urls:
            yield MediaRef(source_type=SourceType.URL, source_uri=url)
