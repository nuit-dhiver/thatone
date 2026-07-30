"""Content and perceptual hashing.

Two hashes with different jobs:

* :func:`content_hash` — sha256 of the raw bytes. This is *identity*: the same
  file from a local path and a URL is one item, described once.
* :func:`dhash` — a 64-bit perceptual hash. This is *similarity*: it survives
  re-encoding and rescaling, so it catches a re-uploaded copy that hashes
  differently, and it drives adaptive frame sampling by measuring how much the
  picture actually changed between frames.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

DHASH_SIZE = 8
"""Edge length of the reduced image. 8 gives a 64-bit hash: small enough to
store as an integer, detailed enough to separate genuinely different frames."""

CHUNK_SIZE = 1024 * 1024


def content_hash(data: bytes) -> str:
    """sha256 of raw bytes, hex-encoded."""
    return hashlib.sha256(data).hexdigest()


def content_hash_file(path: str | Path) -> str:
    """sha256 of a file, streamed so large videos never land in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, size: int = DHASH_SIZE) -> int:
    """Difference hash: compare each pixel to its right-hand neighbour.

    Chosen over average-hash because it keys on *gradients* rather than
    absolute brightness, so a fade or a global exposure shift does not read as
    a scene change — which matters when the hash is deciding whether a frame is
    worth paying a vision model to look at.

    Returns a ``size * size`` bit integer (64 bits at the default size).
    """
    reduced = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    # tobytes() rather than getdata(): for mode "L" it is exactly one byte per
    # pixel in row-major order, it is faster, and it sidesteps getdata()'s
    # deprecation in Pillow 12 without needing a version check.
    pixels = reduced.tobytes()

    bits = 0
    position = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            if pixels[offset + col] > pixels[offset + col + 1]:
                bits |= 1 << position
            position += 1
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits — the distance metric for :func:`dhash`."""
    return (a ^ b).bit_count()


def is_similar(a: int, b: int, *, threshold: int = 4) -> bool:
    """Whether two perceptual hashes are within ``threshold`` bits.

    At 64 bits, ~4 is "the same image, re-encoded" and ~12 is "visibly
    different content".
    """
    return hamming(a, b) <= threshold
