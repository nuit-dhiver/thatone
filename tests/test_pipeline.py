"""End-to-end pipeline and search tests.

Runs the real pipeline — decode, sample, describe, chunk, embed, index, search
— against the stub providers. No API key, no network, no cost, and it exercises
the parts that actually break: stage boundaries, dedupe, ranking, and recovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proper_search.config import Settings, StorageSettings
from proper_search.embed.chunking import build_chunks, split_text
from proper_search.embed.providers import StubEmbeddingProvider
from proper_search.errors import MediaError, MediaTooLargeError
from proper_search.indexer import Indexer
from proper_search.models import (
    ChunkKind,
    Confidence,
    Description,
    FrameNote,
    JobKind,
    MediaStatus,
    SearchFilters,
)
from proper_search.search.fusion import collapse_to_best, reciprocal_rank_fusion
from proper_search.search.pipeline import (
    SIGNAL_DENSE,
    SIGNAL_MEDIA_LEXICAL,
    SearchPipeline,
)
from proper_search.store.sqlite.backend import SQLiteBackend
from proper_search.vision.providers.stub import StubVisionProvider

from . import fixtures

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


class TestChunking:
    def _description(self) -> Description:
        return Description(
            narrative="A man sits at a desk. He stands up abruptly. He walks out of frame.",
            on_screen_text="NOPE",
            tags=["man", "desk", "office"],
            frame_notes=[
                FrameNote(t_ms=0, note="A man sits at a desk looking at the camera."),
                FrameNote(t_ms=900, note="He stands up abruptly, knocking the chair back."),
            ],
            confidence=Confidence.HIGH,
        )

    def test_produces_all_three_kinds(self) -> None:
        chunks = build_chunks("m1", self._description())
        kinds = {c.kind for c in chunks}
        assert kinds == {ChunkKind.NARRATIVE, ChunkKind.SCREEN_TEXT, ChunkKind.FRAME}

    def test_frame_chunks_keep_their_timestamps(self) -> None:
        """The timestamp is what lets a hit say *when* the moment happened."""
        frames = [c for c in build_chunks("m1", self._description()) if c.kind is ChunkKind.FRAME]
        assert [c.t_start_ms for c in frames] == [0, 900]

    def test_on_screen_text_is_its_own_chunk(self) -> None:
        """A remembered quote should match a chunk that is only that quote,
        not one sentence buried in a paragraph."""
        chunks = build_chunks("m1", self._description())
        screen = [c for c in chunks if c.kind is ChunkKind.SCREEN_TEXT]
        assert len(screen) == 1 and screen[0].text == "NOPE"

    def test_frame_chunks_carry_tag_context(self) -> None:
        """A bare 'He stands up.' embeds identically for every clip with a
        person in it; the tags make the vector discriminative."""
        frames = [c for c in build_chunks("m1", self._description()) if c.kind is ChunkKind.FRAME]
        assert "man" in frames[0].text and "desk" in frames[0].text

    def test_ordinals_are_contiguous(self) -> None:
        chunks = build_chunks("m1", self._description())
        assert [c.ord for c in chunks] == list(range(len(chunks)))

    def test_no_chunks_without_a_description(self) -> None:
        empty = Description(narrative="", on_screen_text="", tags=[], frame_notes=[])
        assert build_chunks("m1", empty) == []

    def test_long_narrative_is_split_on_sentences(self) -> None:
        sentences = " ".join(f"This is sentence number {i}." for i in range(40))
        pieces = split_text(sentences, max_chars=200)
        assert len(pieces) > 1
        assert all(len(p) <= 200 for p in pieces)
        assert all(p.endswith(".") for p in pieces), "sentences should stay whole"

    def test_a_single_oversized_sentence_is_hard_wrapped(self) -> None:
        pieces = split_text("word " * 200, max_chars=100)
        assert pieces and all(len(p) <= 100 for p in pieces)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


class TestFusion:
    def test_agreement_between_signals_wins(self) -> None:
        fused = reciprocal_rank_fusion(
            {"a": ["x", "y", "z"], "b": ["y", "x", "z"]}, k=60
        )
        assert fused[0].key in {"x", "y"}
        assert fused[-1].key == "z", "ranked last by both signals"

    def test_top_of_one_signal_beats_middle_of_both(self) -> None:
        """RRF's defining property: a single strong opinion carries weight."""
        fused = reciprocal_rank_fusion(
            {"a": ["strong"] + [f"f{i}" for i in range(20)],
             "b": [f"g{i}" for i in range(10)] + ["mid"]},
            k=10,
        )
        keys = [f.key for f in fused]
        assert keys.index("strong") < keys.index("mid")

    def test_missing_from_a_signal_contributes_zero_not_a_penalty(self) -> None:
        fused = reciprocal_rank_fusion({"a": ["only"], "b": ["other"]}, k=60)
        assert {f.key for f in fused} == {"only", "other"}
        assert fused[0].score == fused[1].score

    def test_ranks_are_reported_for_explainability(self) -> None:
        fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y"]}, k=60)
        by_key = {f.key: f for f in fused}
        assert by_key["y"].ranks == {"a": 2, "b": 1}

    def test_ordering_is_deterministic_on_ties(self) -> None:
        """Wobbling result order makes eval numbers untrustworthy."""
        rankings = {"a": ["p", "q"], "b": ["q", "p"]}
        assert [f.key for f in reciprocal_rank_fusion(rankings)] == [
            f.key for f in reciprocal_rank_fusion(rankings)
        ]

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion({}) == []

    def test_weights_are_applied(self) -> None:
        fused = reciprocal_rank_fusion(
            {"weak": ["a"], "strong": ["b"]}, weights={"strong": 10.0}
        )
        assert fused[0].key == "b"

    def test_collapse_keeps_the_best_chunk_not_the_average(self) -> None:
        """A clip should rank on its single best moment; averaging would
        penalise a long description where one instant matches perfectly."""
        ranked = collapse_to_best([("m1", 0.1), ("m1", 0.9), ("m2", 0.5)])
        assert ranked == ["m1", "m2"]

    def test_collapse_handles_distance_semantics(self) -> None:
        ranked = collapse_to_best(
            [("m1", 0.9), ("m2", 0.1), ("m2", 0.8)], higher_is_better=False
        )
        assert ranked == ["m2", "m1"]


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):
    """A fully wired pipeline backed by stub providers."""
    settings = Settings(
        storage=StorageSettings(path=tmp_path / "index.db", blob_dir=tmp_path / "blobs"),
        vision={"provider": "stub", "model": "stub-vision-1"},
        embedding={"provider": "stub", "model": "stub-embed-1", "dimensions": 64},
        sampling={"min_frames": 2, "max_frames": 6},
    )
    store = SQLiteBackend(settings.storage)
    store.initialize()
    vision = StubVisionProvider(settings=settings.vision)
    embedder = StubEmbeddingProvider(model="stub-embed-1", dimensions=64)
    indexer = Indexer(settings, store, vision, embedder)
    searcher = SearchPipeline(store, embedder, settings.search)
    yield type(
        "Env",
        (),
        {
            "settings": settings, "store": store, "vision": vision,
            "embedder": embedder, "indexer": indexer, "searcher": searcher,
            "tmp": tmp_path,
        },
    )
    store.close()


class TestIngest:
    async def test_ingest_records_media_and_queues_description(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12, size=(240, 135))
        item, created = await env.indexer.ingest_path(path)

        assert created is True
        assert item.mime == "image/gif"
        assert (item.width, item.height) == (240, 135)
        assert item.frame_count == 12
        assert item.duration_ms > 0
        assert item.status is MediaStatus.SAMPLED
        assert len(item.id) == 64

        jobs = env.store.list_jobs(kind=JobKind.DESCRIBE)
        assert len(jobs) == 1 and jobs[0].media_id == item.id

    async def test_thumbnails_are_written(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=10)
        item, _ = await env.indexer.ingest_path(path)
        assert item.thumbnail_path and Path(item.thumbnail_path).exists()
        preview = Path(item.thumbnail_path).with_suffix("").with_suffix(".preview.webp")
        assert preview.exists()

    async def test_same_bytes_under_two_names_is_one_item(self, env, tmp_path: Path) -> None:
        """Identity is the content hash, so a duplicate never costs a second
        vision call."""
        original = fixtures.reaction_gif(tmp_path / "a.gif", frames=8)
        copy = tmp_path / "b.gif"
        copy.write_bytes(original.read_bytes())

        first, created_first = await env.indexer.ingest_path(original)
        second, created_second = await env.indexer.ingest_path(copy)

        assert created_first is True and created_second is False
        assert first.id == second.id
        assert env.store.stats().media_total == 1
        assert len(env.store.get_media(first.id).sources) == 2

    async def test_oversized_file_is_rejected_terminally(self, env, tmp_path: Path) -> None:
        env.settings.fetch.max_bytes = 100
        path = fixtures.reaction_gif(tmp_path / "big.gif", frames=20)
        with pytest.raises(MediaTooLargeError):
            await env.indexer.ingest_path(path)

    async def test_overlong_clip_is_rejected(self, env, tmp_path: Path) -> None:
        env.settings.fetch.max_duration_ms = 100
        path = fixtures.static_gif(tmp_path / "long.gif", frames=60)
        with pytest.raises(MediaTooLargeError):
            await env.indexer.ingest_path(path)

    async def test_missing_file(self, env, tmp_path: Path) -> None:
        with pytest.raises(MediaError):
            await env.indexer.ingest_path(tmp_path / "nope.gif")


class TestDescribeAndEmbed:
    async def test_full_pipeline_reaches_indexed(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.run_all(item.id)

        final = env.store.get_media(item.id)
        assert final.status is MediaStatus.INDEXED
        assert final.vision_model == "stub-vision-1"
        assert final.prompt_version
        assert final.indexed_at is not None

        assert env.store.get_description(item.id) is not None
        stats = env.store.stats()
        assert stats.chunk_total > 0
        assert stats.vector_total == stats.chunk_total, "every chunk must get a vector"

    async def test_describe_sends_the_sampled_frames(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=12)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.describe(item.id)

        assert len(env.vision.calls) == 1, "single_call must issue exactly one request"
        sent = env.vision.calls[0].frames
        assert 2 <= len(sent) <= 6, "frame count must respect the sampling bounds"
        assert all(f.image_bytes.startswith(b"\xff\xd8") for f in sent)

    async def test_describe_queues_the_embed_stage(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=8)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.describe(item.id)
        assert [j.media_id for j in env.store.list_jobs(kind=JobKind.EMBED)] == [item.id]

    async def test_embed_binds_the_vector_space(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=8)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.run_all(item.id)
        stats = env.store.stats()
        assert stats.embedding_model == "stub-embed-1"
        assert stats.embedding_dim == 64

    async def test_deleted_source_gives_an_actionable_error(self, env, tmp_path: Path) -> None:
        path = fixtures.reaction_gif(tmp_path / "gone.gif", frames=8)
        item, _ = await env.indexer.ingest_path(path)
        path.unlink()
        with pytest.raises(MediaError, match="Re-ingest"):
            await env.indexer.describe(item.id)

    async def test_batch_indexing_survives_one_bad_file(self, env, tmp_path: Path) -> None:
        """One corrupt file must not end a 100k-item run."""
        good = [
            fixtures.reaction_gif(tmp_path / f"g{i}.gif", frames=8 + i) for i in range(3)
        ]
        bad = tmp_path / "bad.gif"
        bad.write_bytes(b"GIF89a definitely not a gif")

        results = await env.indexer.index_paths([*good, bad])
        assert len(results) == 3
        assert all(r.status is MediaStatus.INDEXED for r in results)


class TestSearch:
    async def _index(self, env, tmp_path: Path, count: int = 6) -> list[str]:
        paths = [
            fixtures.reaction_gif(tmp_path / f"clip{i}.gif", frames=6 + i * 2)
            for i in range(count)
        ]
        items = await env.indexer.index_paths(paths)
        return [i.id for i in items]

    async def test_finds_an_indexed_clip_by_its_own_words(self, env, tmp_path: Path) -> None:
        ids = await self._index(env, tmp_path, count=5)
        target = ids[0]
        description = env.store.get_description(target)
        query = " ".join(description.narrative.split()[:8])

        hits, diagnostics = await env.searcher.search(query, limit=5)
        assert hits, "an indexed clip must be findable by its own description"
        assert target in {h.media.id for h in hits}
        assert diagnostics.candidates_by_signal

    async def test_both_signals_contribute(self, env, tmp_path: Path) -> None:
        await self._index(env, tmp_path, count=4)
        _, diagnostics = await env.searcher.search("man desk office", limit=5)
        assert SIGNAL_MEDIA_LEXICAL in diagnostics.candidates_by_signal
        assert SIGNAL_DENSE in diagnostics.candidates_by_signal

    async def test_exact_caption_recall(self, env, tmp_path: Path) -> None:
        """The 'I remember it said NOPE' case, which is why on-screen text is
        both its own chunk and a weighted BM25 column."""
        path = fixtures.reaction_gif(tmp_path / "c.gif", frames=8)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.describe(item.id)
        # Force a known caption, then re-embed so the index reflects it.
        description = env.store.get_description(item.id)
        description.on_screen_text = "ABSOLUTELY NOT TODAY"
        env.store.save_description(
            item.id, description, model="m", strategy="s", prompt_version="v1"
        )
        await env.indexer.embed(item.id)

        hits, _ = await env.searcher.search('"absolutely not today"', limit=5)
        assert hits and hits[0].media.id == item.id

    async def test_results_carry_a_snippet_and_timestamp(self, env, tmp_path: Path) -> None:
        ids = await self._index(env, tmp_path, count=3)
        description = env.store.get_description(ids[0])
        hits, _ = await env.searcher.search(description.frame_notes[-1].note, limit=5)
        assert hits
        assert any(h.snippet for h in hits)
        assert any(h.snippet_t_ms is not None for h in hits), (
            "a frame-chunk match should report when the moment happened"
        )

    async def test_explain_reports_per_signal_ranks(self, env, tmp_path: Path) -> None:
        await self._index(env, tmp_path, count=3)
        hits, _ = await env.searcher.search("man office desk", limit=3, explain=True)
        assert hits and hits[0].signals

    async def test_filters_narrow_results(self, env, tmp_path: Path) -> None:
        ids = await self._index(env, tmp_path, count=4)
        tag = env.store.get_description(ids[0]).tags[0]
        hits, _ = await env.searcher.search(
            "clip", limit=10, filters=SearchFilters(tags=[tag])
        )
        for hit in hits:
            assert tag in env.store.get_description(hit.media.id).tags

    async def test_empty_query_returns_nothing(self, env) -> None:
        hits, diagnostics = await env.searcher.search("   ")
        assert hits == [] and diagnostics.fused_count == 0

    async def test_nonsense_query_returns_no_results_rather_than_erroring(
        self, env, tmp_path: Path
    ) -> None:
        await self._index(env, tmp_path, count=2)
        hits, _ = await env.searcher.search("zzzqqqxxx", limit=5)
        assert isinstance(hits, list)

    async def test_hostile_query_does_not_raise(self, env, tmp_path: Path) -> None:
        await self._index(env, tmp_path, count=2)
        for hostile in ['" OR "', "NOT AND NEAR/2", "col:val*", "-(a OR b)"]:
            await env.searcher.search(hostile, limit=5)

    async def test_search_degrades_to_lexical_when_embedding_fails(
        self, env, tmp_path: Path
    ) -> None:
        """Half a result set beats an error page."""
        await self._index(env, tmp_path, count=3)

        class BrokenEmbedder(StubEmbeddingProvider):
            async def embed(self, texts, *, input_type=None):  # type: ignore[override]
                from proper_search.errors import ProviderUnavailable

                raise ProviderUnavailable("embedding provider is down")

        degraded = SearchPipeline(
            env.store, BrokenEmbedder(dimensions=64), env.settings.search
        )
        hits, diagnostics = await degraded.search("man office", limit=5)
        assert diagnostics.degraded_signals and SIGNAL_DENSE in diagnostics.degraded_signals
        assert hits, "lexical results should still come back"

    async def test_deleted_media_disappears_from_results(self, env, tmp_path: Path) -> None:
        ids = await self._index(env, tmp_path, count=3)
        query = env.store.get_description(ids[0]).narrative
        assert ids[0] in {h.media.id for h in (await env.searcher.search(query, limit=5))[0]}

        env.store.delete_media(ids[0])
        hits, _ = await env.searcher.search(query, limit=5)
        assert ids[0] not in {h.media.id for h in hits}


class TestRerank:
    async def test_rerank_runs_and_preserves_the_result_set(self, env, tmp_path: Path) -> None:
        paths = [fixtures.reaction_gif(tmp_path / f"r{i}.gif", frames=6 + i) for i in range(4)]
        await env.indexer.index_paths(paths)

        env.settings.search.rerank = "llm"
        reranker = StubVisionProvider(model="stub-rerank")
        searcher = SearchPipeline(
            env.store, env.embedder, env.settings.search, reranker=reranker
        )
        hits, diagnostics = await searcher.search("man in an office", limit=4)

        assert diagnostics.reranked is True
        assert hits
        assert reranker.calls, "the reranker should have been consulted"
        assert all(len(c.frames) == 0 for c in reranker.calls), (
            "reranking scores text; re-sending frames would be pure waste"
        )

    async def test_rerank_failure_does_not_lose_results(self, env, tmp_path: Path) -> None:
        from proper_search.errors import ProviderUnavailable

        paths = [fixtures.reaction_gif(tmp_path / f"r{i}.gif", frames=6 + i) for i in range(3)]
        await env.indexer.index_paths(paths)

        env.settings.search.rerank = "llm"
        broken = StubVisionProvider(fail_with=ProviderUnavailable("reranker down"))
        searcher = SearchPipeline(env.store, env.embedder, env.settings.search, reranker=broken)
        hits, _ = await searcher.search("man in an office", limit=3)
        assert hits, "a reranker outage must not empty the result set"


class TestResumability:
    async def test_interrupted_run_resumes_without_redoing_work(
        self, env, tmp_path: Path
    ) -> None:
        """The core resumability guarantee: a crash between stages costs
        nothing already paid for."""
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=10)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.describe(item.id)
        calls_after_describe = len(env.vision.calls)

        # Simulate a crash after describe: a fresh indexer picks up the queue.
        resumed = Indexer(env.settings, env.store, env.vision, env.embedder)
        await resumed.embed(item.id)

        assert len(env.vision.calls) == calls_after_describe, (
            "resuming must not re-run the vision call that was already paid for"
        )
        assert env.store.get_media(item.id).status is MediaStatus.INDEXED

    async def test_reingest_of_an_indexed_item_does_not_requeue_description(
        self, env, tmp_path: Path
    ) -> None:
        path = fixtures.reaction_gif(tmp_path / "r.gif", frames=8)
        item, _ = await env.indexer.ingest_path(path)
        await env.indexer.run_all(item.id)

        job = env.store.list_jobs(kind=JobKind.DESCRIBE)[0]
        env.store.complete_job(job.id)

        await env.indexer.ingest_path(path)
        assert env.store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == [], (
            "re-scanning an indexed corpus must not re-bill the vision model"
        )


class TestEvalHarness:
    """The harness that stands in for a UI when judging relevance."""

    async def _indexed(self, env, tmp_path: Path) -> list[str]:
        paths = [
            fixtures.reaction_gif(tmp_path / f"e{i}.gif", frames=6 + i * 2) for i in range(5)
        ]
        return [i.id for i in await env.indexer.index_paths(paths)]

    async def test_scores_each_signal_separately(self, env, tmp_path: Path) -> None:
        """Reporting signals in isolation is what shows whether fusion earns
        its keep, rather than just that search works."""
        from proper_search.eval import EvalHarness, GoldenQuery

        ids = await self._indexed(env, tmp_path)
        golden = [
            GoldenQuery(
                query=env.store.get_description(mid).narrative,
                expected=[mid],
                note="exact narrative",
            )
            for mid in ids
        ]

        results = await EvalHarness(env.store, env.embedder, env.settings.search).run(golden)
        names = {r.name for r in results}
        assert names == {"lexical only", "dense only", "fused"}
        for result in results:
            assert result.query_count == len(golden)
            assert set(result.recall_at) == {1, 5, 10}

    async def test_a_clip_is_findable_by_its_own_description(self, env, tmp_path: Path) -> None:
        from proper_search.eval import EvalHarness, GoldenQuery

        ids = await self._indexed(env, tmp_path)
        golden = [
            GoldenQuery(query=env.store.get_description(mid).narrative, expected=[mid])
            for mid in ids
        ]
        results = await EvalHarness(env.store, env.embedder, env.settings.search).run(golden)
        fused = next(r for r in results if r.name == "fused")
        assert fused.recall_at[10] == 1.0, f"misses: {[m.query for m in fused.misses]}"

    async def test_unanswerable_queries_are_reported_as_misses(self, env, tmp_path: Path) -> None:
        from proper_search.eval import EvalHarness, GoldenQuery, format_report

        await self._indexed(env, tmp_path)
        golden = [GoldenQuery(query="zzz qqq xxx", expected=["nonexistent"], note="impossible")]
        results = await EvalHarness(env.store, env.embedder, env.settings.search).run(golden)
        fused = next(r for r in results if r.name == "fused")
        assert fused.recall_at[10] == 0.0
        assert len(fused.misses) == 1

        report = format_report(results)
        assert "impossible" in report, "the report should say what failed, not just how much"

    async def test_dense_configs_are_skipped_without_an_embedder(self, env, tmp_path: Path) -> None:
        """Skipping beats reporting a zero, which would read as a regression
        rather than a missing provider."""
        from proper_search.eval import EvalHarness, GoldenQuery

        ids = await self._indexed(env, tmp_path)
        golden = [GoldenQuery(query="man office", expected=[ids[0]])]
        results = await EvalHarness(env.store, None, env.settings.search).run(golden)
        assert {r.name for r in results} == {"lexical only"}

    async def test_golden_set_roundtrips_through_json(self, tmp_path: Path) -> None:
        from proper_search.eval import GoldenQuery, load_golden_set, save_golden_set

        original = [
            GoldenQuery(query="the guy who slowly turns around", expected=["a" * 64], note="vague")
        ]
        path = tmp_path / "golden.json"
        save_golden_set(original, path)
        loaded = load_golden_set(path)
        assert loaded[0].query == original[0].query
        assert loaded[0].expected == original[0].expected
        assert loaded[0].note == "vague"

    async def test_empty_golden_set(self, env) -> None:
        from proper_search.eval import EvalHarness, format_report

        results = await EvalHarness(env.store, env.embedder, env.settings.search).run([])
        assert results == []
        assert "empty golden set" in format_report(results)
