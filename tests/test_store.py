"""Storage backend tests."""

from __future__ import annotations

import time

import pytest

from proper_search.errors import IndexConsistencyError, StorageError
from proper_search.models import (
    Chunk,
    ChunkKind,
    JobKind,
    JobState,
    MediaSource,
    MediaStatus,
    SearchFilters,
    SourceType,
)
from proper_search.store.sqlite.backend import SQLiteBackend, build_fts_query

from .conftest import make_description, make_media

# --------------------------------------------------------------------------
# FTS query construction
# --------------------------------------------------------------------------


class TestBuildFtsQuery:
    def test_plain_words_are_or_joined_and_quoted(self) -> None:
        assert build_fts_query("cat jumping") == '"cat" OR "jumping"'

    def test_quoted_phrase_is_preserved_as_a_phrase(self) -> None:
        assert build_fts_query('he said "not today"') == '"not today" OR "he" OR "said"'

    def test_empty_and_punctuation_only_return_none(self) -> None:
        # Callers skip the lexical leg entirely on None rather than issuing a
        # query that matches everything.
        assert build_fts_query("") is None
        assert build_fts_query("   ") is None
        assert build_fts_query("!!! ??? ***") is None

    @pytest.mark.parametrize(
        "hostile",
        [
            'cat" OR "" OR "',  # quote-breaking
            "cat AND NOT dog",  # bare operators
            "cat NEAR/5 dog",  # NEAR syntax
            "col:value",  # column filter syntax
            "cat*",  # prefix wildcard
            "^cat",  # initial-token operator
            "(cat OR dog) AND bird",  # grouping
            "-cat",  # negation
            'a" OR media_fts MATCH "b',
        ],
    )
    def test_operator_syntax_in_user_input_is_neutralized(
        self, store: SQLiteBackend, hostile: str
    ) -> None:
        """User text must never be parsed as FTS5 syntax.

        The strong assertion is not the string shape but that the query
        *executes*: an unescaped operator raises OperationalError, which in a
        search endpoint is a 500 on ordinary user input.
        """
        media = make_media("h")
        store.upsert_media(media)
        store.save_description(
            media.id, make_description(), model="m", strategy="s", prompt_version="v1"
        )
        store.search_lexical(hostile, limit=10)  # must not raise

    def test_stemming_matches_inflections(self, store: SQLiteBackend) -> None:
        media = make_media("s")
        store.upsert_media(media)
        store.save_description(
            media.id,
            make_description(narrative="A man walks away from the desk."),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        # Porter stemming is why "walking" finds "walks".
        assert store.search_lexical("walking", limit=5)


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------


class TestMedia:
    def test_initialize_is_idempotent(self, store: SQLiteBackend) -> None:
        store.initialize()
        store._initialized = False  # force the schema script to run again
        store.initialize()
        assert store.stats().media_total == 0

    def test_upsert_reports_created_then_updated(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        assert store.upsert_media(media) is True
        assert store.upsert_media(media) is False, "same content hash must not create a second row"
        assert store.stats().media_total == 1

    def test_same_content_from_two_sources_is_one_item(self, store: SQLiteBackend) -> None:
        """Identity is the content hash, so a re-host does not get described twice."""
        media = make_media("a")
        store.upsert_media(media)
        store.add_source(
            media.id,
            MediaSource(source_type=SourceType.URL, source_uri="https://example.com/a.gif"),
        )
        fetched = store.get_media(media.id)
        assert fetched is not None
        assert len(fetched.sources) == 2
        assert store.stats().media_total == 1
        assert store.find_by_source_uri("https://example.com/a.gif") == media.id

    def test_update_preserves_created_at_and_provenance(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id,
            make_description(),
            model="claude-sonnet-5",
            strategy="single_call",
            prompt_version="v1",
        )
        # A bare re-scan carries no provenance; it must not erase what is stored.
        store.upsert_media(make_media("a"))
        again = store.get_media(media.id)
        assert again is not None
        assert again.vision_model == "claude-sonnet-5"
        assert again.prompt_version == "v1"
        assert again.created_at == media.created_at

    def test_get_media_many_batches(self, store: SQLiteBackend) -> None:
        ids = []
        for seed in "abcde":
            item = make_media(seed)
            store.upsert_media(item)
            ids.append(item.id)
        found = store.get_media_many([*ids, "missing"])
        assert set(found) == set(ids)

    def test_near_duplicate_detection_by_hamming_distance(self, store: SQLiteBackend) -> None:
        base = 0b1010101010101010
        store.upsert_media(make_media("a", phash=base))
        store.upsert_media(make_media("b", phash=base ^ 0b11))  # 2 bits away
        store.upsert_media(make_media("c", phash=base ^ 0xFFFF))  # far away

        near = store.find_near_duplicates(base, max_distance=4)
        assert len(near) == 2, "a re-encode of the same GIF should be caught as a near-duplicate"
        assert len(store.find_near_duplicates(base, max_distance=0)) == 1

    def test_delete_cascades(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id, make_description(), model="m", strategy="s", prompt_version="v1"
        )
        store.replace_chunks(
            media.id, [Chunk(media_id=media.id, ord=0, kind=ChunkKind.NARRATIVE, text="x")]
        )
        store.enqueue(media.id, JobKind.EMBED)

        assert store.delete_media(media.id) is True
        assert store.get_media(media.id) is None
        assert store.get_chunks(media.id) == []
        assert store.get_description(media.id) is None
        assert store.list_jobs() == []
        assert store.search_lexical("man", limit=5) == []


# --------------------------------------------------------------------------
# Descriptions
# --------------------------------------------------------------------------


class TestDescriptions:
    def test_roundtrip(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        desc = make_description(on_screen_text="NOPE")
        store.save_description(
            media.id, desc, model="m", strategy="single_call", prompt_version="v1"
        )

        loaded = store.get_description(media.id)
        assert loaded is not None
        assert loaded.narrative == desc.narrative
        assert loaded.on_screen_text == "NOPE"
        assert loaded.tags == desc.tags
        assert len(loaded.frame_notes) == 3
        assert loaded.frame_notes[1].t_ms == 900

    def test_status_advances_to_described(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id, make_description(), model="m", strategy="s", prompt_version="v1"
        )
        item = store.get_media(media.id)
        assert item is not None and item.status is MediaStatus.DESCRIBED

    def test_redescribe_replaces_tags_rather_than_merging(self, store: SQLiteBackend) -> None:
        """A dropped tag must actually disappear, or filters drift from the text."""
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id,
            make_description(tags=["cat", "kitchen"]),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        store.save_description(
            media.id,
            make_description(tags=["dog"]),
            model="m",
            strategy="s",
            prompt_version="v2",
        )
        assert store.filter_media_ids(SearchFilters(tags=["cat"])) == set()
        assert store.filter_media_ids(SearchFilters(tags=["dog"])) == {media.id}

    def test_redescribe_replaces_lexical_index(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id,
            make_description(narrative="A parrot on a bicycle."),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        assert store.search_lexical("parrot", limit=5)

        store.save_description(
            media.id,
            make_description(narrative="A tortoise on a skateboard."),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        assert store.search_lexical("parrot", limit=5) == [], "stale text must not stay searchable"
        assert store.search_lexical("tortoise", limit=5)


# --------------------------------------------------------------------------
# Chunks and vectors
# --------------------------------------------------------------------------


class TestChunksAndVectors:
    def _seed(self, store: SQLiteBackend, seed: str = "a") -> tuple[str, list[int]]:
        media = make_media(seed)
        store.upsert_media(media)
        chunks = [
            Chunk(media_id=media.id, ord=0, kind=ChunkKind.NARRATIVE, text="whole clip summary"),
            Chunk(
                media_id=media.id,
                ord=1,
                kind=ChunkKind.FRAME,
                text="he stands up",
                t_start_ms=900,
                t_end_ms=900,
            ),
            Chunk(media_id=media.id, ord=2, kind=ChunkKind.SCREEN_TEXT, text="NOPE"),
        ]
        return media.id, store.replace_chunks(media.id, chunks)

    def test_replace_chunks_assigns_ids_and_indexes_text(self, store: SQLiteBackend) -> None:
        media_id, ids = self._seed(store)
        assert len(ids) == 3
        assert [c.ord for c in store.get_chunks(media_id)] == [0, 1, 2]
        hits = store.search_chunks_lexical("stands", limit=5)
        assert hits and hits[0].media_id == media_id

    def test_replace_chunks_clears_previous(self, store: SQLiteBackend) -> None:
        media_id, _ = self._seed(store)
        store.replace_chunks(
            media_id, [Chunk(media_id=media_id, ord=0, kind=ChunkKind.NARRATIVE, text="new")]
        )
        assert len(store.get_chunks(media_id)) == 1
        assert store.search_chunks_lexical("stands", limit=5) == []

    def test_vector_roundtrip_and_knn_ordering(self, store: SQLiteBackend) -> None:
        _, ids = self._seed(store)
        store.ensure_vector_space(model="stub", dimensions=4)
        store.save_embeddings(
            [
                (ids[0], [1.0, 0.0, 0.0, 0.0]),
                (ids[1], [0.0, 1.0, 0.0, 0.0]),
                (ids[2], [0.0, 0.0, 1.0, 0.0]),
            ]
        )
        hits = store.search_dense([0.0, 0.9, 0.1, 0.0], limit=3)
        assert hits[0].chunk_id == ids[1], "nearest vector must rank first"
        assert hits[0].distance < hits[1].distance

    def test_saving_a_vector_twice_replaces_it(self, store: SQLiteBackend) -> None:
        _, ids = self._seed(store)
        store.ensure_vector_space(model="stub", dimensions=4)
        store.save_embeddings([(ids[0], [1.0, 0.0, 0.0, 0.0])])
        store.save_embeddings([(ids[0], [0.0, 1.0, 0.0, 0.0])])
        assert store.stats().vector_total == 1

    def test_mismatched_embedding_model_is_refused(self, store: SQLiteBackend) -> None:
        """Mixing embedding spaces yields confident nonsense, so it must raise."""
        store.ensure_vector_space(model="text-embedding-3-small", dimensions=1536)
        with pytest.raises(IndexConsistencyError, match="not comparable"):
            store.ensure_vector_space(model="voyage-3", dimensions=1024)
        with pytest.raises(IndexConsistencyError):
            store.ensure_vector_space(model="text-embedding-3-small", dimensions=512)

    def test_wrong_width_vector_is_refused(self, store: SQLiteBackend) -> None:
        _, ids = self._seed(store)
        store.ensure_vector_space(model="stub", dimensions=4)
        with pytest.raises(IndexConsistencyError, match="dimension"):
            store.save_embeddings([(ids[0], [1.0, 2.0])])
        with pytest.raises(IndexConsistencyError, match="different model"):
            store.search_dense([1.0, 2.0], limit=5)

    def test_querying_before_binding_a_space_is_an_error(self, store: SQLiteBackend) -> None:
        with pytest.raises(StorageError, match="no embedding space"):
            store.search_dense([1.0, 2.0, 3.0], limit=5)

    def test_chunks_missing_embeddings_tracks_backlog(self, store: SQLiteBackend) -> None:
        _, ids = self._seed(store)
        assert len(store.chunks_missing_embeddings()) == 3
        store.ensure_vector_space(model="stub", dimensions=4)
        store.save_embeddings([(ids[0], [1.0, 0.0, 0.0, 0.0])])
        assert {c.id for c in store.chunks_missing_embeddings()} == {ids[1], ids[2]}

    def test_vector_space_survives_reopen(self, store: SQLiteBackend, tmp_path) -> None:
        _, ids = self._seed(store)
        store.ensure_vector_space(model="stub", dimensions=4)
        store.save_embeddings([(ids[0], [1.0, 0.0, 0.0, 0.0])])
        store.close()

        from proper_search.config import StorageSettings

        reopened = SQLiteBackend(StorageSettings(path=store.path))
        reopened.initialize()
        assert reopened.stats().vector_total == 1
        assert reopened.search_dense([1.0, 0.0, 0.0, 0.0], limit=1)[0].chunk_id == ids[0]
        reopened.close()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class TestRetrieval:
    def test_on_screen_text_outranks_narrative_mention(self, store: SQLiteBackend) -> None:
        """A remembered verbatim caption should beat an incidental narrative word."""
        quoted = make_media("quoted")
        store.upsert_media(quoted)
        store.save_description(
            quoted.id,
            make_description(
                narrative="A person reacts to something.", on_screen_text="NOPE", tags=["reaction"]
            ),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        mentioned = make_media("mentioned")
        store.upsert_media(mentioned)
        store.save_description(
            mentioned.id,
            make_description(
                narrative="Someone says nope and shrugs at the camera.",
                on_screen_text="",
                tags=["shrug"],
            ),
            model="m",
            strategy="s",
            prompt_version="v1",
        )

        hits = store.search_lexical("nope", limit=5, weights=(1.0, 3.0, 1.5))
        assert hits[0].media_id == quoted.id

    def test_more_matching_terms_ranks_higher(self, store: SQLiteBackend) -> None:
        both = make_media("both")
        store.upsert_media(both)
        store.save_description(
            both.id,
            make_description(narrative="A cat knocks a glass off a table.", tags=["cat", "glass"]),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        one = make_media("one")
        store.upsert_media(one)
        store.save_description(
            one.id,
            make_description(narrative="A cat sleeps in a sunbeam.", tags=["cat"]),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        hits = store.search_lexical("cat glass table", limit=5)
        assert hits[0].media_id == both.id
        assert {h.media_id for h in hits} == {both.id, one.id}, "OR keeps partial matches"

    def test_snippets_are_returned(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id, make_description(), model="m", strategy="s", prompt_version="v1"
        )
        hits = store.search_lexical("desk", limit=5)
        assert hits and "desk" in hits[0].snippet.lower()


class TestFilters:
    @pytest.fixture
    def populated(self, store: SQLiteBackend) -> SQLiteBackend:
        short = make_media("short", duration_ms=1000)
        store.upsert_media(short)
        store.save_description(
            short.id,
            make_description(
                narrative="A cat in a kitchen.", on_screen_text="", tags=["cat", "kitchen"]
            ),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        long_ = make_media("long", duration_ms=30_000, mime="video/mp4")
        store.upsert_media(long_)
        store.add_source(
            long_.id, MediaSource(source_type=SourceType.URL, source_uri="https://x/y.mp4")
        )
        store.save_description(
            long_.id,
            make_description(
                narrative="A cat on a skateboard.", on_screen_text="WOW", tags=["cat", "skateboard"]
            ),
            model="m",
            strategy="s",
            prompt_version="v1",
        )
        return store

    def test_no_filter_returns_none(self, populated: SQLiteBackend) -> None:
        assert populated.filter_media_ids(SearchFilters()) is None

    def test_tags_use_and_semantics(self, populated: SQLiteBackend) -> None:
        assert len(populated.filter_media_ids(SearchFilters(tags=["cat"])) or set()) == 2
        assert len(populated.filter_media_ids(SearchFilters(tags=["cat", "kitchen"])) or set()) == 1
        assert populated.filter_media_ids(SearchFilters(tags=["cat", "nonexistent"])) == set()

    def test_duration_bounds(self, populated: SQLiteBackend) -> None:
        assert len(populated.filter_media_ids(SearchFilters(max_duration_ms=5000)) or set()) == 1
        assert len(populated.filter_media_ids(SearchFilters(min_duration_ms=5000)) or set()) == 1

    def test_has_on_screen_text(self, populated: SQLiteBackend) -> None:
        with_text = populated.filter_media_ids(SearchFilters(has_on_screen_text=True)) or set()
        without = populated.filter_media_ids(SearchFilters(has_on_screen_text=False)) or set()
        assert len(with_text) == 1 and len(without) == 1
        assert not (with_text & without)

    def test_source_and_mime(self, populated: SQLiteBackend) -> None:
        assert (
            len(populated.filter_media_ids(SearchFilters(source_type=SourceType.URL)) or set()) == 1
        )
        assert len(populated.filter_media_ids(SearchFilters(mime="video/mp4")) or set()) == 1

    def test_filters_apply_to_lexical_search(self, populated: SQLiteBackend) -> None:
        unfiltered = populated.search_lexical("cat", limit=10)
        assert len(unfiltered) == 2
        filtered = populated.search_lexical(
            "cat", limit=10, filters=SearchFilters(tags=["skateboard"])
        )
        assert len(filtered) == 1

    def test_filters_apply_to_dense_search(self, populated: SQLiteBackend) -> None:
        populated.ensure_vector_space(model="stub", dimensions=4)
        for idx, media_id in enumerate(
            sorted(populated.filter_media_ids(SearchFilters(tags=["cat"])) or [])
        ):
            ids = populated.replace_chunks(
                media_id,
                [Chunk(media_id=media_id, ord=0, kind=ChunkKind.NARRATIVE, text="cat")],
            )
            populated.save_embeddings([(ids[0], [float(idx), 1.0, 0.0, 0.0])])

        assert len(populated.search_dense([0.0, 1.0, 0.0, 0.0], limit=10)) == 2
        narrowed = populated.search_dense(
            [0.0, 1.0, 0.0, 0.0], limit=10, filters=SearchFilters(mime="video/mp4")
        )
        assert len(narrowed) == 1


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class TestJobs:
    def test_enqueue_is_idempotent_per_stage(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        first = store.enqueue(media.id, JobKind.DESCRIBE)
        assert store.enqueue(media.id, JobKind.DESCRIBE) == first
        store.enqueue(media.id, JobKind.EMBED)
        assert len(store.list_jobs()) == 2, "stages are tracked independently"

    def test_claim_leases_and_excludes_claimed(self, store: SQLiteBackend) -> None:
        for seed in "abc":
            item = make_media(seed)
            store.upsert_media(item)
            store.enqueue(item.id, JobKind.DESCRIBE)

        claimed = store.claim_jobs(JobKind.DESCRIBE, limit=2, lease_seconds=60)
        assert len(claimed) == 2
        assert all(j.attempts == 1 for j in claimed)

        # A second worker sees only what is left.
        again = store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60)
        assert len(again) == 1
        assert not ({j.media_id for j in claimed} & {j.media_id for j in again})

    def test_claim_filters_by_stage(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.EMBED)
        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == []
        assert len(store.claim_jobs(JobKind.EMBED, limit=5, lease_seconds=60)) == 1

    def test_complete_removes_from_the_pool(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        job = store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=60)[0]
        assert job.id is not None
        store.complete_job(job.id)
        assert store.list_jobs(state="done")[0].media_id == media.id
        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == []

    def test_retryable_failure_returns_to_pending_after_backoff(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        job = store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=60)[0]
        assert job.id is not None

        store.fail_job(job.id, "rate limited", retry=True, backoff_seconds=30)
        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == [], (
            "backoff must actually delay the retry"
        )

        store.fail_job(job.id, "rate limited", retry=True, backoff_seconds=0)
        retried = store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60)
        assert len(retried) == 1
        assert retried[0].attempts == 2

    def test_terminal_failure_is_not_retried(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        job = store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=60)[0]
        assert job.id is not None

        store.fail_job(job.id, "refusal: cyber", retry=False)
        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == []
        failed = store.list_jobs(state=str(JobState.FAILED))
        assert len(failed) == 1 and "refusal" in (failed[0].last_error or "")

    def test_expired_lease_is_reclaimed(self, store: SQLiteBackend) -> None:
        """A killed worker must not strand its in-flight job forever."""
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=-1)  # already expired

        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == []
        assert store.reclaim_expired_leases() == 1
        assert len(store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60)) == 1

    def test_live_lease_is_not_reclaimed(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=600)
        assert store.reclaim_expired_leases() == 0, "a healthy worker must not lose its job"

    def test_requeue_after_failure_resets_state(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        job = store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=60)[0]
        assert job.id is not None
        store.fail_job(job.id, "boom", retry=False)

        store.enqueue(media.id, JobKind.DESCRIBE)
        assert len(store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60)) == 1

    def test_requeue_does_not_redo_completed_work(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.enqueue(media.id, JobKind.DESCRIBE)
        job = store.claim_jobs(JobKind.DESCRIBE, limit=1, lease_seconds=60)[0]
        assert job.id is not None
        store.complete_job(job.id)

        store.enqueue(media.id, JobKind.DESCRIBE)
        assert store.claim_jobs(JobKind.DESCRIBE, limit=5, lease_seconds=60) == [], (
            "re-scanning an indexed corpus must not re-bill the vision model"
        )


class TestConcurrency:
    def test_parallel_workers_never_claim_the_same_job(self, store: SQLiteBackend) -> None:
        """The whole resumability story rests on claims being exclusive."""
        import threading

        for i in range(60):
            item = make_media(f"job{i}")
            store.upsert_media(item)
            store.enqueue(item.id, JobKind.DESCRIBE)

        claimed: list[str] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                own: list[str] = []
                while batch := store.claim_jobs(JobKind.DESCRIBE, limit=4, lease_seconds=60):
                    own.extend(j.media_id for j in batch)
                    time.sleep(0.001)
                with lock:
                    claimed.extend(own)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent claiming raised: {errors}"
        assert len(claimed) == 60
        assert len(set(claimed)) == 60, "a job was claimed by two workers simultaneously"


class TestStats:
    def test_reports_counts_and_bound_embedding_space(self, store: SQLiteBackend) -> None:
        media = make_media("a")
        store.upsert_media(media)
        store.save_description(
            media.id, make_description(), model="m", strategy="s", prompt_version="v1"
        )
        ids = store.replace_chunks(
            media.id, [Chunk(media_id=media.id, ord=0, kind=ChunkKind.NARRATIVE, text="x")]
        )
        store.ensure_vector_space(model="stub-embed", dimensions=4)
        store.save_embeddings([(ids[0], [1.0, 0.0, 0.0, 0.0])])
        store.enqueue(media.id, JobKind.EMBED)

        stats = store.stats()
        assert stats.media_total == 1
        assert stats.chunk_total == 1
        assert stats.vector_total == 1
        assert stats.tag_total == 4
        assert stats.embedding_model == "stub-embed"
        assert stats.embedding_dim == 4
        assert (stats.media_by_status or {})[str(MediaStatus.DESCRIBED)] == 1
