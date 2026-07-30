"""Synthetic media fixtures.

Generated rather than committed as binaries: the interesting properties (frame
counts, scene-change structure, variable delays, optimizer-produced partial
tiles) are then explicit in code and can be tuned per test, instead of being
opaque bytes nobody can adjust.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

BG = (30, 60, 120)
FG = (240, 200, 40)


def _save_gif(
    frames: list[Image.Image],
    path: Path,
    *,
    duration: int | list[int] = 100,
    optimize: bool = True,
) -> Path:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=optimize,
    )
    return path


def reaction_gif(path: Path, *, frames: int = 12, size: tuple[int, int] = (240, 135)) -> Path:
    """A short GIF with several visually distinct scenes.

    Stands in for the common case: a couple of seconds, a handful of real
    beats, the thing worth describing happening somewhere in the middle.
    """
    palette = [(200, 40, 40), (40, 160, 80), (60, 60, 200), (220, 180, 40)]
    images = []
    for i in range(frames):
        scene = palette[(i * len(palette)) // frames]
        img = Image.new("RGB", size, scene)
        draw = ImageDraw.Draw(img)
        draw.ellipse([20 + i * 12, 40, 70 + i * 12, 90], fill=(250, 250, 250))
        images.append(img)
    return _save_gif(images, path)


def static_gif(path: Path, *, frames: int = 60, size: tuple[int, int] = (160, 90)) -> Path:
    """Many frames, almost no change.

    The case fixed-interval sampling handles worst: naive sampling bills for
    dozens of near-identical frames that tell the model nothing new.
    """
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, BG)
        # Sub-perceptual drift only — nowhere near a scene-change threshold.
        ImageDraw.Draw(img).rectangle([10, 10, 60, 40], fill=(FG[0], FG[1], FG[2] + i % 3))
        images.append(img)
    return _save_gif(images, path)


def optimized_gif(path: Path, *, frames: int = 8, size: tuple[int, int] = (200, 100)) -> Path:
    """A GIF the encoder stores as partial tiles with disposal methods.

    A small square moves across a static background, so Pillow's optimizer
    writes frames 1..n as sub-rectangles rather than full canvases. Decoding
    must composite them back onto the previous frame; a decoder that does not
    yields torn frames with holes, which degrades descriptions silently rather
    than failing loudly.
    """
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, BG)
        ImageDraw.Draw(img).rectangle([10 + i * 20, 30, 30 + i * 20, 60], fill=FG)
        images.append(img)
    return _save_gif(images, path, optimize=True)


def single_frame_gif(path: Path, *, size: tuple[int, int] = (120, 120)) -> Path:
    """A one-frame GIF — effectively a still image.

    Sampling must not divide by zero or demand ``min_frames`` it cannot get.
    """
    img = Image.new("RGB", size, BG)
    ImageDraw.Draw(img).ellipse([20, 20, 100, 100], fill=FG)
    img.save(path, format="GIF")
    return path


def variable_delay_gif(path: Path, *, size: tuple[int, int] = (120, 80)) -> Path:
    """Per-frame delays of 20ms, 500ms, 20ms, 1000ms.

    Timestamps must come from real presentation times: an ``index / fps``
    estimate puts every frame in the wrong place on a GIF like this, and the
    frame timestamps are what let a hit say *when* the moment happened.
    """
    durations = [20, 500, 20, 1000]
    images = []
    for i in range(4):
        img = Image.new("RGB", size, (40 + i * 50, 40, 40))
        ImageDraw.Draw(img).rectangle([10, 10, 50 + i * 10, 50], fill=FG)
        images.append(img)
    return _save_gif(images, path, duration=durations)


def gradient_gif(path: Path, *, frames: int = 20, size: tuple[int, int] = (160, 90)) -> Path:
    """A steady sweep with no hard cuts — change accumulates gradually.

    Scene-change detection has no obvious boundary to latch onto here, so this
    exercises the accumulate-since-last-kept behaviour rather than
    change-since-previous-frame.
    """
    images = []
    for i in range(frames):
        shade = int(255 * i / max(1, frames - 1))
        img = Image.new("RGB", size, (shade, shade // 2, 255 - shade))
        ImageDraw.Draw(img).rectangle([i * 6, 20, i * 6 + 30, 60], fill=(255 - shade, 255, shade))
        images.append(img)
    return _save_gif(images, path)


def mp4_clip(path: Path, *, frames: int = 30, size: tuple[int, int] = (160, 96)) -> Path:
    """A tiny H.264 MP4.

    Proves the "video is a config change, not a rewrite" claim with an actual
    non-GIF container rather than an assertion in a docstring.
    """
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        # Deterministic output, and avoids libx264 lookahead reordering frames.
        stream.options = {"crf": "28", "preset": "ultrafast", "tune": "zerolatency"}

        for i in range(frames):
            img = Image.new("RGB", size, (20, 20 + (i * 7) % 200, 60))
            ImageDraw.Draw(img).rectangle([i * 4, 20, i * 4 + 24, 60], fill=FG)
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
