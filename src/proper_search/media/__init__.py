"""Media decoding, sampling, hashing, and thumbnails."""

from __future__ import annotations

from .decode import DecodedFrame, decode_at_indices, encode_frame, iter_frames, probe
from .hashing import content_hash, content_hash_file, dhash, hamming, is_similar
from .sampling import (
    AdaptiveSampler,
    CountSampler,
    FrameMeta,
    IntervalSampler,
    SamplingStrategy,
    get_strategy,
    sample_frames,
    scan_timeline,
)
from .thumbnail import encode_poster_bytes, write_poster, write_preview

__all__ = [  # noqa: RUF022
    # Decoding
    "DecodedFrame",
    "probe",
    "iter_frames",
    "decode_at_indices",
    "encode_frame",
    # Hashing
    "content_hash",
    "content_hash_file",
    "dhash",
    "hamming",
    "is_similar",
    # Sampling
    "FrameMeta",
    "SamplingStrategy",
    "AdaptiveSampler",
    "IntervalSampler",
    "CountSampler",
    "get_strategy",
    "sample_frames",
    "scan_timeline",
    # Thumbnails
    "write_poster",
    "write_preview",
    "encode_poster_bytes",
]
