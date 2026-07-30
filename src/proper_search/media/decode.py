"""Frame decoding, via PyAV (FFmpeg).

PyAV rather than Pillow, even though the first target is GIFs:

* **Reach.** The same code path decodes MP4, WebM, and MOV, so extending from
  GIFs to video is a configuration change rather than a second decoder with its
  own bugs. This is the main reason.
* **Timestamps.** GIF frame delays vary per frame, so an ``index / fps``
  estimate drifts. FFmpeg exposes real presentation timestamps, which is what
  makes "the part at 0:03" mean anything.
* **Threaded decode** for the video formats, where it matters.

Note on optimized GIFs: they store most frames as partial tiles plus a disposal
method. Both FFmpeg and modern Pillow composite these correctly on *sequential*
iteration — this was measured, not assumed. It is only random-access seeking
that gets fragile, which is why :func:`iter_frames` never seeks.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

from ..errors import DecodeError, UnsupportedMediaError
from ..models import MediaProbe

# FFmpeg reports a container format name, which is often a comma-joined list of
# everything the demuxer handles. Mapped to a single stable MIME per format.
_FORMAT_MIME = {
    "gif": "image/gif",
    "webp": "image/webp",
    "apng": "image/apng",
    "png_pipe": "image/png",
    "mov,mp4,m4a,3gp,3g2,mj2": "video/mp4",
    "matroska,webm": "video/webm",
    "avi": "video/x-msvideo",
    "mpegts": "video/mp2t",
    "flv": "video/x-flv",
}

DEFAULT_FPS = 10.0
"""Fallback frame rate when the container declares none — common in GIFs, whose
per-frame delays make an average rate meaningless."""


@dataclass(slots=True)
class DecodedFrame:
    index: int
    t_ms: int
    image: Image.Image


def _mime_for(format_name: str) -> str:
    return _FORMAT_MIME.get(format_name, f"video/{format_name.split(',')[0]}")


def _open(path: str | Path) -> av.container.InputContainer:
    try:
        return av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        # FFmpegError is not an OSError subclass in PyAV, and a missing file
        # can surface as either depending on the protocol handler.
        raise DecodeError(f"could not open {path}: {exc}") from exc


def probe(path: str | Path) -> MediaProbe:
    """Read container-level facts without decoding pixels.

    Everything here is best-effort: GIF containers routinely report a frame
    count of zero and a nonsense average rate. The authoritative counts come
    from :func:`iter_frames`, which is why callers update the item after
    sampling rather than trusting this alone.
    """
    with _open(path) as container:
        if not container.streams.video:
            raise UnsupportedMediaError(f"{path} contains no video stream")
        stream = container.streams.video[0]

        # FFmpeg opens a file on container magic alone, so a truncated or
        # corrupt download yields a stream with 0x0 dimensions rather than an
        # error. Caught here, or it becomes a stored media row with no pixels
        # that only fails much later, during description.
        if not stream.codec_context.width or not stream.codec_context.height:
            raise DecodeError(
                f"{path} has a video stream with no dimensions; the file is "
                f"probably truncated or corrupt"
            )

        duration_ms = 0
        if stream.duration is not None and stream.time_base:
            duration_ms = int(stream.duration * stream.time_base * 1000)
        elif container.duration is not None:
            duration_ms = int(container.duration / 1000)  # AV_TIME_BASE is microseconds

        rate: Fraction | None = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate and rate > 0 else 0.0

        frame_count = stream.frames or 0
        if frame_count == 0 and fps > 0 and duration_ms > 0:
            frame_count = max(1, round(duration_ms / 1000 * fps))

        return MediaProbe(
            mime=_mime_for(container.format.name),
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            duration_ms=max(0, duration_ms),
            frame_count=frame_count,
            fps=fps if fps > 0 else DEFAULT_FPS,
            is_animated=frame_count != 1,
        )


def iter_frames(path: str | Path, *, max_frames: int = 3000) -> Iterator[DecodedFrame]:
    """Decode frames sequentially.

    Sequential rather than seeking: seeking is unreliable across GIF and
    variable-frame-rate containers, and these files are short enough that a
    linear read is cheap and always correct.

    ``max_frames`` is a hard ceiling so a pathological input cannot spin
    forever or exhaust memory during ingest.
    """
    with _open(path) as container:
        if not container.streams.video:
            raise UnsupportedMediaError(f"{path} contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        time_base = stream.time_base
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate and rate > 0 else DEFAULT_FPS

        index = 0
        try:
            for frame in container.decode(stream):
                if index >= max_frames:
                    return
                # pts is authoritative: GIF frame delays vary per frame, so
                # index/fps drifts badly on anything non-uniform.
                if frame.pts is not None and time_base:
                    t_ms = int(frame.pts * time_base * 1000)
                else:
                    t_ms = int(index / fps * 1000)
                # PyAV ships partial stubs; to_image() is untyped but returns Image.
                image = frame.to_image()  # type: ignore[no-untyped-call]
                yield DecodedFrame(index=index, t_ms=max(0, t_ms), image=image)
                index += 1
        except av.FFmpegError as exc:
            # A truncated file that decoded some frames is still usable; only a
            # failure before any frame is fatal.
            if index == 0:
                raise DecodeError(f"could not decode {path}: {exc}") from exc


def decode_at_indices(
    path: str | Path, indices: Sequence[int], *, max_frames: int = 3000
) -> list[DecodedFrame]:
    """Decode only the frames at ``indices``, in a single sequential pass."""
    wanted = set(indices)
    if not wanted:
        return []
    limit = min(max_frames, max(wanted) + 1)
    out: list[DecodedFrame] = []
    for frame in iter_frames(path, max_frames=limit):
        if frame.index in wanted:
            out.append(frame)
            if len(out) == len(wanted):
                break
    return out


def encode_frame(
    image: Image.Image, *, max_edge: int = 768, quality: int = 85
) -> tuple[bytes, int, int]:
    """Downscale and JPEG-encode a frame for transmission.

    Downscaling here is the dominant cost control: vision providers bill images
    by area (roughly ``width * height / 750`` tokens), so halving the longest
    edge quarters the per-frame token cost. 768px preserves enough detail for
    burned-in caption text, which is the hardest thing the model has to read.

    Returns ``(jpeg_bytes, width, height)``.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / longest
        # At least 1px per side: a 2000x1 sliver would otherwise scale to zero.
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), width, height
