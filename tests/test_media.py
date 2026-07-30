"""Media layer tests: decoding, hashing, sampling, thumbnails."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from thatone.config import SamplingSettings
from thatone.errors import DecodeError
from thatone.media.decode import (
    decode_at_indices,
    encode_frame,
    iter_frames,
    probe,
)
from thatone.media.hashing import content_hash, content_hash_file, dhash, hamming, is_similar
from thatone.media.sampling import (
    AdaptiveSampler,
    CountSampler,
    FrameMeta,
    IntervalSampler,
    get_strategy,
    sample_frames,
    scan_timeline,
)
from thatone.media.thumbnail import write_poster, write_preview

from . import fixtures

# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


class TestHashing:
    def test_content_hash_is_stable_and_distinguishing(self) -> None:
        assert content_hash(b"abc") == content_hash(b"abc")
        assert content_hash(b"abc") != content_hash(b"abd")
        assert len(content_hash(b"abc")) == 64

    def test_file_hash_matches_bytes_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        payload = b"x" * (3 * 1024 * 1024)  # spans several stream chunks
        path.write_bytes(payload)
        assert content_hash_file(path) == content_hash(payload)

    def test_dhash_is_64_bits(self) -> None:
        img = Image.new("RGB", (64, 64), (10, 20, 30))
        ImageDraw.Draw(img).rectangle([0, 0, 31, 63], fill=(200, 200, 200))
        assert 0 <= dhash(img) < 2**64

    def test_identical_images_hash_identically(self) -> None:
        img = Image.new("RGB", (64, 64), (10, 20, 30))
        ImageDraw.Draw(img).ellipse([10, 10, 50, 50], fill=(200, 100, 50))
        assert dhash(img) == dhash(img.copy())

    def test_rescaling_barely_moves_the_hash(self) -> None:
        """The point of a perceptual hash: survive re-encoding and resizing."""
        img = Image.new("RGB", (200, 200), (30, 30, 30))
        ImageDraw.Draw(img).ellipse([40, 40, 160, 160], fill=(220, 180, 40))
        shrunk = img.resize((100, 100), Image.Resampling.LANCZOS)
        assert is_similar(dhash(img), dhash(shrunk), threshold=6)

    def test_different_images_hash_far_apart(self) -> None:
        left = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(left).rectangle([0, 0, 31, 63], fill=(255, 255, 255))
        right = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(right).rectangle([0, 0, 63, 31], fill=(255, 255, 255))
        assert hamming(dhash(left), dhash(right)) > 8

    def test_uniform_brightness_shift_is_not_a_scene_change(self) -> None:
        """dHash keys on gradients, so a global exposure change must not read
        as new content — otherwise every fade burns frames on the vision bill."""
        base = Image.new("RGB", (64, 64), (60, 60, 60))
        ImageDraw.Draw(base).rectangle([10, 10, 50, 50], fill=(160, 160, 160))
        brighter = Image.eval(base, lambda v: min(255, v + 40))
        assert hamming(dhash(base), dhash(brighter)) <= 2


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


class TestDecode:
    def test_probe_reads_gif_geometry(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12, size=(240, 135))
        info = probe(path)
        assert info.mime == "image/gif"
        assert (info.width, info.height) == (240, 135)
        assert info.duration_ms > 0

    def test_decodes_every_frame(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        frames = list(iter_frames(path))
        assert len(frames) == 12
        assert [f.index for f in frames] == list(range(12))

    def test_max_frames_caps_decoding(self, tmp_path: Path) -> None:
        path = fixtures.static_gif(tmp_path / "s.gif", frames=60)
        assert len(list(iter_frames(path, max_frames=5))) == 5

    def test_optimized_gif_frames_are_composited(self, tmp_path: Path) -> None:
        """Partial-tile frames must come back as full composited canvases.

        The fixture's frames 1..n are stored as small sub-rectangles. If a
        decoder returned those tiles uncomposited, the untouched background
        would be black or transparent instead of the fixture's blue — a silent
        quality failure that would poison every description.
        """
        path = fixtures.optimized_gif(tmp_path / "o.gif", frames=8, size=(200, 100))
        frames = list(iter_frames(path))
        assert len(frames) == 8
        for frame in frames:
            assert frame.image.size == (200, 100), "frame is a partial tile, not a full canvas"
            corner = frame.image.convert("RGB").getpixel((2, 2))
            assert corner == pytest.approx(fixtures.BG, abs=8), (
                f"background lost at frame {frame.index}: got {corner}"
            )

    def test_variable_delays_produce_nonuniform_timestamps(self, tmp_path: Path) -> None:
        """Timestamps must track real per-frame delays, not index/fps."""
        path = fixtures.variable_delay_gif(tmp_path / "v.gif")
        stamps = [f.t_ms for f in iter_frames(path)]
        assert stamps[0] == 0
        gaps = [b - a for a, b in pairwise(stamps)]
        assert max(gaps) > 3 * min(gaps), f"delays look uniform: {gaps}"

    def test_single_frame_gif_decodes(self, tmp_path: Path) -> None:
        path = fixtures.single_frame_gif(tmp_path / "one.gif")
        assert len(list(iter_frames(path))) == 1

    def test_decode_at_indices_returns_only_requested(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        frames = decode_at_indices(path, [0, 5, 11])
        assert sorted(f.index for f in frames) == [0, 5, 11]

    def test_corrupt_file_raises_decode_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.gif"
        bad.write_bytes(b"GIF89a this is not actually a gif")
        with pytest.raises(DecodeError):
            probe(bad)

    def test_missing_file_raises_decode_error(self, tmp_path: Path) -> None:
        with pytest.raises(DecodeError):
            probe(tmp_path / "nope.gif")

    def test_mp4_decodes_through_the_same_path(self, tmp_path: Path) -> None:
        """The video expansion is a config change, not a second decoder."""
        path = fixtures.mp4_clip(tmp_path / "clip.mp4", frames=30)
        info = probe(path)
        assert info.mime == "video/mp4"
        assert (info.width, info.height) == (160, 96)
        assert len(list(iter_frames(path))) >= 25


class TestEncodeFrame:
    def test_downscales_to_max_edge(self) -> None:
        payload, width, height = encode_frame(Image.new("RGB", (1600, 900)), max_edge=768)
        assert max(width, height) == 768
        assert height == 432, "aspect ratio must be preserved"
        assert payload.startswith(b"\xff\xd8"), "expected a JPEG"

    def test_small_frames_are_not_upscaled(self) -> None:
        _, width, height = encode_frame(Image.new("RGB", (120, 80)), max_edge=768)
        assert (width, height) == (120, 80)

    def test_extreme_aspect_ratio_keeps_at_least_one_pixel(self) -> None:
        """A 2000x1 sliver must not scale to a zero-height image."""
        _, width, height = encode_frame(Image.new("RGB", (2000, 1)), max_edge=100)
        assert width == 100 and height >= 1

    def test_max_edge_drives_payload_size(self) -> None:
        img = Image.new("RGB", (1200, 1200), (10, 20, 30))
        ImageDraw.Draw(img).ellipse([100, 100, 1100, 1100], fill=(200, 150, 60))
        assert len(encode_frame(img, max_edge=256)[0]) < len(encode_frame(img, max_edge=1024)[0])


# --------------------------------------------------------------------------
# Sampling strategies (pure, no media required)
# --------------------------------------------------------------------------


def timeline(*hashes: int, step_ms: int = 100) -> list[FrameMeta]:
    return [FrameMeta(index=i, t_ms=i * step_ms, phash=h) for i, h in enumerate(hashes)]


class TestAdaptiveSampler:
    settings = SamplingSettings(min_frames=1, max_frames=100, hamming_threshold=12)

    def test_keeps_the_first_frame(self) -> None:
        picked = AdaptiveSampler().select(timeline(0, 0, 0), self.settings)
        assert picked[0].index == 0

    def test_drops_visually_identical_frames(self) -> None:
        picked = AdaptiveSampler().select(timeline(0b1111, 0b1111, 0b1111), self.settings)
        assert len(picked) == 1, "identical frames must not each cost a vision call"

    def test_keeps_frames_past_the_threshold(self) -> None:
        far = (1 << 20) - 1  # 20 bits set: well past a threshold of 12
        picked = AdaptiveSampler().select(timeline(0, far, 0), self.settings)
        assert [p.index for p in picked] == [0, 1, 2]

    def test_compares_against_last_kept_not_last_seen(self) -> None:
        """Gradual drift must accumulate.

        Comparing only to the immediately preceding frame means a slow pan
        never trips the threshold and the whole clip collapses to one frame.
        """
        gradual = timeline(0b0, 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111)
        settings = SamplingSettings(min_frames=1, max_frames=100, hamming_threshold=4)
        picked = AdaptiveSampler().select(gradual, settings)
        assert len(picked) >= 2, "accumulated drift should eventually register as a change"

    def test_min_frames_tops_up_a_static_clip(self) -> None:
        settings = SamplingSettings(min_frames=3, max_frames=12, hamming_threshold=12)
        picked = AdaptiveSampler().select(timeline(*([7] * 20)), settings)
        assert len(picked) == 3
        assert [p.index for p in picked] == sorted(p.index for p in picked)

    def test_min_frames_cannot_exceed_available_frames(self) -> None:
        settings = SamplingSettings(min_frames=8, max_frames=12, hamming_threshold=12)
        picked = AdaptiveSampler().select(timeline(1, 2), settings)
        assert len(picked) == 2, "a 2-frame GIF cannot yield 8 frames"

    def test_max_frames_is_a_hard_ceiling(self) -> None:
        settings = SamplingSettings(min_frames=1, max_frames=5, hamming_threshold=1)
        picked = AdaptiveSampler().select(timeline(*[1 << i for i in range(40)]), settings)
        assert len(picked) == 5, "cost ceiling must hold even on high-motion clips"

    def test_capping_spreads_across_the_clip(self) -> None:
        """Truncating to the first N frames would describe only the opening."""
        settings = SamplingSettings(min_frames=1, max_frames=4, hamming_threshold=1)
        picked = AdaptiveSampler().select(timeline(*[1 << i for i in range(40)]), settings)
        assert picked[-1].index >= 30, f"selection stopped early: {[p.index for p in picked]}"

    def test_empty_timeline(self) -> None:
        assert AdaptiveSampler().select([], self.settings) == []


class TestIntervalSampler:
    def test_picks_one_frame_per_interval(self) -> None:
        settings = SamplingSettings(strategy="interval", interval_seconds=0.5, max_frames=100,
                                    min_frames=1)
        picked = IntervalSampler().select(timeline(*range(20), step_ms=100), settings)
        assert [p.t_ms for p in picked] == [0, 500, 1000, 1500]

    def test_short_clip_still_yields_a_frame(self) -> None:
        """The failure mode that motivated the adaptive default: a 300ms GIF
        under a 2-second interval must not produce zero frames."""
        settings = SamplingSettings(strategy="interval", interval_seconds=2.0, max_frames=100,
                                    min_frames=1)
        picked = IntervalSampler().select(timeline(*range(3), step_ms=100), settings)
        assert len(picked) == 1

    def test_respects_max_frames(self) -> None:
        settings = SamplingSettings(strategy="interval", interval_seconds=0.1, max_frames=3,
                                    min_frames=1)
        picked = IntervalSampler().select(timeline(*range(50), step_ms=100), settings)
        assert len(picked) == 3


class TestCountSampler:
    def test_returns_evenly_spaced_frames(self) -> None:
        settings = SamplingSettings(strategy="count", target_count=5, max_frames=100, min_frames=1)
        picked = CountSampler().select(timeline(*range(21), step_ms=100), settings)
        assert [p.index for p in picked] == [0, 5, 10, 15, 20]

    def test_covers_the_whole_clip(self) -> None:
        settings = SamplingSettings(strategy="count", target_count=4, max_frames=100, min_frames=1)
        picked = CountSampler().select(timeline(*range(100), step_ms=10), settings)
        assert picked[0].index == 0 and picked[-1].index == 99

    def test_short_clip_returns_what_exists(self) -> None:
        settings = SamplingSettings(strategy="count", target_count=10, max_frames=100, min_frames=1)
        assert len(CountSampler().select(timeline(1, 2, 3), settings)) == 3

    def test_max_frames_overrides_target_count(self) -> None:
        settings = SamplingSettings(strategy="count", target_count=20, max_frames=5, min_frames=1)
        assert len(CountSampler().select(timeline(*range(50)), settings)) == 5


class TestStrategyRegistry:
    @pytest.mark.parametrize("name", ["adaptive", "interval", "count"])
    def test_known_strategies_resolve(self, name: str) -> None:
        assert get_strategy(name).name == name

    def test_unknown_strategy_lists_the_options(self) -> None:
        with pytest.raises(ValueError, match="adaptive"):
            get_strategy("magic")


# --------------------------------------------------------------------------
# End-to-end sampling
# --------------------------------------------------------------------------


class TestSampleFrames:
    def test_produces_encoded_samples(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        samples = sample_frames(path, SamplingSettings(min_frames=2, max_frames=6))
        assert 2 <= len(samples) <= 6
        for sample in samples:
            assert sample.image_bytes.startswith(b"\xff\xd8")
            assert sample.width > 0 and sample.height > 0
            assert sample.media_type == "image/jpeg"

    def test_samples_are_time_ordered(self, tmp_path: Path) -> None:
        path = fixtures.gradient_gif(tmp_path / "g.gif", frames=20)
        samples = sample_frames(path, SamplingSettings(min_frames=3, max_frames=8))
        assert [s.t_ms for s in samples] == sorted(s.t_ms for s in samples)

    def test_static_clip_costs_far_less_than_a_varied_one(self, tmp_path: Path) -> None:
        """The whole point of adaptive sampling, measured end to end."""
        settings = SamplingSettings(min_frames=1, max_frames=30, hamming_threshold=12)
        static = sample_frames(fixtures.static_gif(tmp_path / "s.gif", frames=60), settings)
        varied = sample_frames(fixtures.reaction_gif(tmp_path / "r.gif", frames=30), settings)
        assert len(static) < len(varied), (
            f"static clip sampled {len(static)} frames vs {len(varied)} for a varied one"
        )

    def test_single_frame_gif_yields_one_sample(self, tmp_path: Path) -> None:
        path = fixtures.single_frame_gif(tmp_path / "one.gif")
        samples = sample_frames(path, SamplingSettings(min_frames=3, max_frames=12))
        assert len(samples) == 1

    def test_reuses_a_supplied_timeline(self, tmp_path: Path) -> None:
        """Ingest scans once; sampling must not decode a third time."""
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        scanned = scan_timeline(path)
        samples = sample_frames(path, SamplingSettings(max_frames=4), timeline=scanned)
        assert samples

    def test_frame_max_edge_bounds_every_sample(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=6, size=(1200, 800))
        samples = sample_frames(path, SamplingSettings(max_frames=3, frame_max_edge=128))
        assert all(max(s.width, s.height) <= 128 for s in samples)

    def test_works_on_mp4(self, tmp_path: Path) -> None:
        path = fixtures.mp4_clip(tmp_path / "clip.mp4", frames=30)
        samples = sample_frames(path, SamplingSettings(min_frames=2, max_frames=5))
        assert 2 <= len(samples) <= 5

    def test_scan_timeline_on_empty_input_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.gif"
        bad.write_bytes(b"not a gif at all")
        with pytest.raises(DecodeError):
            scan_timeline(bad)


# --------------------------------------------------------------------------
# Thumbnails
# --------------------------------------------------------------------------


class TestThumbnails:
    def test_poster_is_written_and_sharded(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=6)
        frames = list(iter_frames(path))
        media_id = "ab" + "c" * 62
        poster = write_poster(frames[0].image, tmp_path / "thumbs", media_id)
        assert poster.exists()
        assert poster.parent.name == "ab", "thumbnails must shard, not pile into one directory"
        assert Image.open(poster).format == "WEBP"

    def test_poster_respects_max_edge(self, tmp_path: Path) -> None:
        big = Image.new("RGB", (2000, 1000), (10, 20, 30))
        poster = write_poster(big, tmp_path / "thumbs", "aa" + "b" * 62, max_edge=200)
        assert max(Image.open(poster).size) == 200

    def test_preview_is_animated(self, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=10)
        frames = list(iter_frames(path))
        preview = write_preview(frames, tmp_path / "thumbs", "cd" + "e" * 62)
        assert preview is not None and preview.exists()
        assert getattr(Image.open(preview), "n_frames", 1) > 1

    def test_preview_is_skipped_for_a_still(self, tmp_path: Path) -> None:
        path = fixtures.single_frame_gif(tmp_path / "one.gif")
        frames = list(iter_frames(path))
        assert write_preview(frames, tmp_path / "thumbs", "ef" + "0" * 62) is None

    def test_preview_caps_frame_count(self, tmp_path: Path) -> None:
        path = fixtures.static_gif(tmp_path / "s.gif", frames=60)
        frames = list(iter_frames(path))
        preview = write_preview(frames, tmp_path / "thumbs", "ff" + "1" * 62, max_frames=8)
        assert preview is not None
        assert getattr(Image.open(preview), "n_frames", 1) <= 8
