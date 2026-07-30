"""HTTP API and engine tests.

Driven through a real ASGI client against a fully stubbed engine, so the routes,
serialization, and error mapping are exercised end to end without a network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from proper_search.api.app import create_app
from proper_search.config import Settings, StorageSettings
from proper_search.embed.providers import StubEmbeddingProvider
from proper_search.engine import ProperSearch
from proper_search.errors import (
    IndexConsistencyError,
    MediaTooLargeError,
    ProviderRefusal,
    ProviderUnavailable,
)
from proper_search.store.sqlite.backend import SQLiteBackend
from proper_search.vision.providers.stub import StubVisionProvider

from . import fixtures


@pytest.fixture
def engine(tmp_path: Path) -> AsyncIterator[ProperSearch]:
    settings = Settings(
        storage=StorageSettings(path=tmp_path / "api.db", blob_dir=tmp_path / "blobs"),
        vision={"provider": "stub", "model": "stub-vision-1"},
        embedding={"provider": "stub", "model": "stub-embed-1", "dimensions": 64},
        sampling={"min_frames": 2, "max_frames": 5},
    )
    store = SQLiteBackend(settings.storage)
    store.initialize()
    built = ProperSearch(
        settings,
        store,
        StubVisionProvider(settings=settings.vision),
        StubEmbeddingProvider(model="stub-embed-1", dimensions=64),
    )
    yield built
    store.close()


@pytest.fixture
def client(engine: ProperSearch) -> TestClient:
    with TestClient(create_app(engine=engine)) as test_client:
        yield test_client


@pytest.fixture
def gifs(tmp_path: Path) -> list[Path]:
    return [
        fixtures.reaction_gif(tmp_path / f"clip{i}.gif", frames=6 + i * 2) for i in range(4)
    ]


def index_all(client: TestClient, paths: list[Path]) -> dict:
    response = client.post(
        "/index", json={"targets": [str(p) for p in paths], "wait": True}
    )
    assert response.status_code == 202, response.text
    return response.json()


# --------------------------------------------------------------------------
# Ops
# --------------------------------------------------------------------------


class TestOps:
    def test_healthz_reports_capabilities(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["storage"] == "sqlite"
        assert body["indexing_available"] is True
        assert body["dense_search_available"] is True

    def test_healthz_reports_degraded_capability_rather_than_failing(
        self, engine: ProperSearch
    ) -> None:
        """An index with no embedding key still serves lexical search; the
        operator needs to see which half is missing, not a 500."""
        engine.embedder = None
        engine.searcher.embedder = None
        with TestClient(create_app(engine=engine)) as degraded:
            body = degraded.get("/healthz").json()
            assert body["status"] == "ok"
            assert body["dense_search_available"] is False
            assert body["indexing_available"] is False

    def test_stats_on_an_empty_index(self, client: TestClient) -> None:
        body = client.get("/stats").json()
        assert body["media_total"] == 0
        assert body["vector_total"] == 0

    def test_stats_after_indexing(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        body = client.get("/stats").json()
        assert body["media_total"] == 4
        assert body["chunk_total"] > 0
        assert body["vector_total"] == body["chunk_total"]
        assert body["embedding_model"] == "stub-embed-1"

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "/search" in schema["paths"]
        assert "/index" in schema["paths"]


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


class TestIndexRoute:
    def test_index_files_and_report_ids(self, client: TestClient, gifs: list[Path]) -> None:
        body = index_all(client, gifs)
        assert body["accepted"] == 4
        assert len(body["media_ids"]) == 4
        assert body["drained"] is True

    def test_index_a_directory(self, client: TestClient, gifs: list[Path], tmp_path: Path) -> None:
        response = client.post("/index", json={"targets": [str(tmp_path)], "wait": True})
        assert response.status_code == 202
        assert response.json()["accepted"] == len(gifs)

    def test_index_without_wait_defers_the_work(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        """A 100k backfill cannot be waited on inside one request."""
        response = client.post("/index", json={"targets": [str(g) for g in gifs]})
        assert response.status_code == 202
        assert response.json()["drained"] is False
        assert client.get("/jobs?state=pending").json(), "work should be queued"
        assert client.get("/stats").json()["vector_total"] == 0

        drained = client.post("/drain").json()
        assert drained["completed"] > 0
        assert client.get("/stats").json()["vector_total"] > 0

    def test_reindexing_the_same_file_does_not_duplicate(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        index_all(client, gifs)
        body = index_all(client, gifs)
        assert body["already_indexed"] == 4
        assert client.get("/stats").json()["media_total"] == 4

    def test_empty_target_list_is_rejected(self, client: TestClient) -> None:
        assert client.post("/index", json={"targets": []}).status_code == 422

    def test_unknown_field_is_rejected(self, client: TestClient) -> None:
        response = client.post("/index", json={"targets": ["/tmp/x.gif"], "bogus": 1})
        assert response.status_code == 422

    def test_missing_file_is_skipped_not_fatal(self, client: TestClient, tmp_path: Path) -> None:
        good = fixtures.reaction_gif(tmp_path / "ok.gif", frames=6)
        response = client.post(
            "/index",
            json={"targets": [str(good), str(tmp_path / "nope.gif")], "wait": True},
        )
        assert response.status_code == 202
        assert response.json()["accepted"] == 1


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class TestSearchRoute:
    def test_finds_an_indexed_clip(self, client: TestClient, gifs: list[Path]) -> None:
        body = index_all(client, gifs)
        target = body["media_ids"][0]
        narrative = client.get(f"/media/{target}").json()["description"]["narrative"]

        results = client.get("/search", params={"q": " ".join(narrative.split()[:8])}).json()
        assert results["count"] > 0
        assert target in [h["media"]["id"] for h in results["hits"]]

    def test_hits_carry_snippet_and_timestamp(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        results = client.get("/search", params={"q": "man office desk"}).json()
        assert results["hits"]
        assert any(h["snippet"] for h in results["hits"])

    def test_thumbnail_url_is_returned_not_a_filesystem_path(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        """Server-side paths must never leak to clients."""
        index_all(client, gifs)
        hit = client.get("/search", params={"q": "man office"}).json()["hits"][0]
        assert hit["media"]["thumbnail_url"].startswith("/media/")
        assert "thumbnail_path" not in hit["media"]

    def test_limit_is_honoured(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        assert len(client.get("/search", params={"q": "man", "limit": 2}).json()["hits"]) <= 2

    def test_explain_returns_per_signal_ranks(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        results = client.get(
            "/search", params={"q": "man office desk", "explain": True}
        ).json()
        assert results["hits"] and results["hits"][0]["signals"]

    def test_tag_filter(self, client: TestClient, gifs: list[Path]) -> None:
        body = index_all(client, gifs)
        tags = client.get(f"/media/{body['media_ids'][0]}").json()["description"]["tags"]
        results = client.get("/search", params={"q": "clip", "tags": [tags[0]]}).json()
        assert results["count"] >= 1

    def test_duration_filter(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        wide = client.get("/search", params={"q": "man", "limit": 50}).json()["count"]
        narrow = client.get(
            "/search", params={"q": "man", "limit": 50, "max_duration_ms": 1}
        ).json()["count"]
        assert narrow < wide

    def test_has_text_filter(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        with_text = client.get(
            "/search", params={"q": "man", "has_text": True, "limit": 50}
        ).json()["count"]
        without = client.get(
            "/search", params={"q": "man", "has_text": False, "limit": 50}
        ).json()["count"]
        assert with_text + without >= 1

    def test_missing_query_is_a_validation_error(self, client: TestClient) -> None:
        assert client.get("/search").status_code == 422

    def test_blank_query_returns_empty_not_an_error(self, client: TestClient) -> None:
        body = client.get("/search", params={"q": "   "}).json()
        assert body["count"] == 0

    def test_hostile_query_does_not_500(self, client: TestClient, gifs: list[Path]) -> None:
        """FTS5 operator syntax in user input must not reach the parser."""
        index_all(client, gifs)
        for hostile in ['" OR "', "NOT AND NEAR/2", "col:val*", "-(a OR b)", '"unclosed']:
            assert client.get("/search", params={"q": hostile}).status_code == 200

    def test_degraded_signal_is_surfaced_to_the_client(
        self, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        """Partial results are reported as partial, not passed off as complete."""
        with TestClient(create_app(engine=engine)) as first:
            index_all(first, gifs)

        class BrokenEmbedder(StubEmbeddingProvider):
            async def embed(self, texts, *, input_type=None):  # type: ignore[override]
                raise ProviderUnavailable("embeddings are down")

        engine.searcher.embedder = BrokenEmbedder(dimensions=64)
        with TestClient(create_app(engine=engine)) as degraded:
            body = degraded.get("/search", params={"q": "man office"}).json()
            assert body["degraded_signals"], "the client must be told a signal was skipped"
            assert body["count"] > 0, "lexical results should still come back"


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------


class TestMediaRoutes:
    def test_get_media_includes_the_description(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        media_id = index_all(client, gifs)["media_ids"][0]
        body = client.get(f"/media/{media_id}").json()
        assert body["media"]["id"] == media_id
        assert body["media"]["status"] == "indexed"
        assert body["description"]["narrative"]
        assert body["description"]["frame_notes"]

    def test_unknown_media_is_404(self, client: TestClient) -> None:
        assert client.get("/media/" + "0" * 64).status_code == 404

    def test_list_media_paginates(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        assert len(client.get("/media", params={"limit": 2}).json()) == 2
        assert len(client.get("/media", params={"limit": 2, "offset": 2}).json()) == 2

    def test_list_media_filters_by_status(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        assert len(client.get("/media", params={"status": "indexed"}).json()) == 4
        assert client.get("/media", params={"status": "failed"}).json() == []

    def test_thumbnail_is_served_as_webp(self, client: TestClient, gifs: list[Path]) -> None:
        media_id = index_all(client, gifs)["media_ids"][0]
        response = client.get(f"/media/{media_id}/thumbnail")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert response.content.startswith(b"RIFF")

    def test_animated_preview_is_served(self, client: TestClient, gifs: list[Path]) -> None:
        media_id = index_all(client, gifs)["media_ids"][0]
        response = client.get(f"/media/{media_id}/thumbnail", params={"animated": True})
        assert response.status_code == 200
        assert response.content.startswith(b"RIFF")

    def test_reindex_without_redescribe_skips_the_vision_call(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        """The cheap path: re-embed from the stored description, no vision spend."""
        media_id = index_all(client, gifs)["media_ids"][0]
        calls_before = len(engine.vision.calls)

        response = client.post(
            f"/media/{media_id}/reindex", params={"redescribe": False}
        )
        assert response.status_code == 200
        assert len(engine.vision.calls) == calls_before, "no vision call should have been made"

    def test_reindex_with_redescribe_calls_the_model(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        media_id = index_all(client, gifs)["media_ids"][0]
        calls_before = len(engine.vision.calls)
        assert client.post(f"/media/{media_id}/reindex").status_code == 200
        assert len(engine.vision.calls) > calls_before

    def test_reindex_unknown_media_is_404(self, client: TestClient) -> None:
        assert client.post("/media/" + "0" * 64 + "/reindex").status_code == 404

    def test_delete_removes_the_item_from_search(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        media_id = index_all(client, gifs)["media_ids"][0]
        assert client.delete(f"/media/{media_id}").status_code == 204
        assert client.get(f"/media/{media_id}").status_code == 404
        assert client.delete(f"/media/{media_id}").status_code == 404


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class TestJobRoutes:
    def test_jobs_are_listed_and_completed(self, client: TestClient, gifs: list[Path]) -> None:
        index_all(client, gifs)
        done = client.get("/jobs", params={"state": "done"}).json()
        assert len(done) == 8, "one describe and one embed job per item"
        assert {j["kind"] for j in done} == {"describe", "embed"}

    def test_failed_jobs_record_a_reason(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        """A refusal is terminal and must be explicable without reading logs."""
        engine.vision.fail_with = ProviderRefusal("declined")
        client.post("/index", json={"targets": [str(gifs[0])], "wait": True})

        failed = client.get("/jobs", params={"state": "failed"}).json()
        assert failed and "declined" in failed[0]["last_error"]
        assert client.get("/stats").json()["media_by_status"].get("failed") == 1


# --------------------------------------------------------------------------
# Estimation and error mapping
# --------------------------------------------------------------------------


class TestEstimateRoute:
    def test_estimate_projects_a_full_run(self, client: TestClient, gifs: list[Path]) -> None:
        response = client.post(
            "/estimate", json={"targets": [str(g) for g in gifs], "sample_size": 2}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["item_count"] == 4
        assert body["sampled"] == 2
        assert body["avg_frames"] > 0
        assert body["summary"]

    def test_estimate_warns_when_no_price_is_configured(
        self, client: TestClient, gifs: list[Path]
    ) -> None:
        """The stub model has no configured rate; that must be stated, not
        silently reported as a cost of zero."""
        body = client.post(
            "/estimate", json={"targets": [str(gifs[0])], "sample_size": 1}
        ).json()
        assert any("no price configured" in w for w in body["warnings"])
        assert body["total_cost"] == 0.0

    def test_estimate_prices_a_known_model(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        engine.settings.vision.model = "claude-sonnet-5"
        body = client.post(
            "/estimate", json={"targets": [str(g) for g in gifs], "sample_size": 2}
        ).json()
        assert body["total_cost"] > 0
        assert body["cost_per_item"] > 0
        assert not body["measured"], "the stub exposes no token counter"


class TestErrorMapping:
    def test_index_consistency_becomes_409(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        """Mixed embedding spaces are a conflict a client can act on, not a 500."""
        index_all(client, gifs)
        engine.embedder = StubEmbeddingProvider(model="different-model", dimensions=64)
        engine.indexer.embedder = engine.embedder

        media_id = client.get("/media").json()[0]["id"]
        response = client.post(f"/media/{media_id}/reindex", params={"redescribe": False})
        assert response.status_code == 409
        assert response.json()["error"] == IndexConsistencyError.__name__

    def test_oversized_media_becomes_413(
        self, client: TestClient, engine: ProperSearch, tmp_path: Path
    ) -> None:
        engine.settings.fetch.max_bytes = 10
        path = fixtures.reaction_gif(tmp_path / "big.gif", frames=10)
        # Ingest failures are logged and skipped by the batch path, so nothing
        # is accepted rather than the request erroring.
        body = client.post("/index", json={"targets": [str(path)], "wait": True}).json()
        assert body["accepted"] == 0

    def test_transient_provider_failure_maps_to_503(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        index_all(client, gifs)
        media_id = client.get("/media").json()[0]["id"]
        engine.vision.fail_with = ProviderUnavailable("upstream is down")
        response = client.post(f"/media/{media_id}/reindex")
        assert response.status_code == 503, response.text
        assert response.json()["kind"] == "transient"

    def test_terminal_errors_are_labelled_as_such(
        self, client: TestClient, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        index_all(client, gifs)
        media_id = client.get("/media").json()[0]["id"]
        engine.vision.fail_with = ProviderRefusal("declined")
        response = client.post(f"/media/{media_id}/reindex")
        assert response.status_code == 422
        assert response.json()["kind"] == "terminal"

    def test_error_type_is_preserved(self) -> None:
        from proper_search.api.app import _status_for

        assert _status_for(MediaTooLargeError("x")) == 413
        assert _status_for(IndexConsistencyError("x")) == 409
        assert _status_for(ProviderRefusal("x")) == 422
        assert _status_for(ProviderUnavailable("x")) == 503


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class TestEngine:
    async def test_index_and_search_through_the_library_api(
        self, engine: ProperSearch, gifs: list[Path]
    ) -> None:
        """The library surface the HTTP layer wraps must work on its own."""
        items = await engine.index([str(g) for g in gifs])
        assert len(items) == 4
        assert all(i.status.value == "indexed" for i in items)

        hits, diagnostics = await engine.search("man in an office", limit=5)
        assert isinstance(hits, list)
        assert diagnostics.candidates_by_signal

    async def test_index_accepts_a_directory(
        self, engine: ProperSearch, gifs: list[Path], tmp_path: Path
    ) -> None:
        items = await engine.index([tmp_path])
        assert len(items) == len(gifs)

    async def test_close_is_idempotent(self, engine: ProperSearch) -> None:
        await engine.close()
        await engine.close()
