"""SQLite storage backend: FTS5 for lexical, sqlite-vec for dense.

One file, no server, and it holds 100k items comfortably. Connections are
thread-local because async callers dispatch storage work through
``asyncio.to_thread`` and a single SQLite connection is not safely shared
across threads.

WAL mode plus a non-zero ``busy_timeout`` are what make concurrent ingest
workable: without them, routine write contention surfaces as spurious
``database is locked`` errors rather than a short wait.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ...config import StorageSettings
from ...errors import (
    ExtensionUnavailableError,
    IndexConsistencyError,
    SchemaVersionError,
    StorageError,
)
from ...models import (
    Chunk,
    ChunkKind,
    Confidence,
    Description,
    FrameNote,
    Job,
    JobKind,
    JobState,
    MediaItem,
    MediaSource,
    MediaStatus,
    SearchFilters,
    SourceType,
    utcnow,
)
from ..base import DenseMatch, IndexStats, LexicalMatch, StorageBackend

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

META_SCHEMA_VERSION = "schema_version"
META_EMBEDDING_MODEL = "embedding_model"
META_EMBEDDING_DIM = "embedding_dim"

VECTOR_TABLE = "chunk_vectors"

# FTS5 treats a pile of characters as operators (AND OR NOT NEAR * ^ - : " ()).
# Every user token is quoted before it reaches MATCH, so a query like
# 'the one where he says "nope" -- NOT sure' is searched literally instead of
# being parsed as syntax or raising.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_PHRASE_RE = re.compile(r'"([^"]*)"')


def build_fts_query(raw: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression.

    Quoted spans stay phrases; everything else becomes OR-ed terms. OR rather
    than AND is deliberate for this use case — someone recalling a GIF gets
    details wrong, and one bad term should lower the rank, not empty the result
    set. BM25 already rewards documents that match more terms.

    Returns None when nothing searchable survives, so callers can skip the
    lexical leg entirely instead of issuing a query that matches everything.
    """
    parts: list[str] = []
    for phrase in _PHRASE_RE.findall(raw):
        cleaned = " ".join(_TOKEN_RE.findall(phrase))
        if cleaned:
            parts.append(f'"{cleaned}"')
    remainder = _PHRASE_RE.sub(" ", raw)
    parts.extend(f'"{token}"' for token in _TOKEN_RE.findall(remainder))
    if not parts:
        return None
    return " OR ".join(parts)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class SQLiteBackend(StorageBackend):
    """SQLite implementation of :class:`~proper_search.store.base.StorageBackend`."""

    def __init__(self, settings: StorageSettings) -> None:
        self.settings = settings
        self.path = Path(settings.path).expanduser()
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self._vector_dim: int | None = None

    # -- connection --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path,
            timeout=self.settings.busy_timeout_ms / 1000.0,
            isolation_level=None,  # explicit transactions; no implicit BEGIN
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self.settings.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._load_vec_extension(conn)
        return conn

    @staticmethod
    def _load_vec_extension(conn: sqlite3.Connection) -> None:
        """Load sqlite-vec, or explain precisely why vector search is impossible.

        Some Python distributions are built without ``enable_load_extension``.
        That is unfixable at runtime, so the error names the cause and the fix
        rather than surfacing later as a missing-function error deep in a query.
        """
        if not hasattr(conn, "enable_load_extension"):
            raise ExtensionUnavailableError(
                "this Python's sqlite3 was compiled without extension support, so "
                "sqlite-vec cannot be loaded and dense search is unavailable. Use a "
                "Python built with --enable-loadable-sqlite-extensions (Homebrew and "
                "uv-managed builds are), or switch storage.backend to postgres."
            )
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ExtensionUnavailableError(f"could not load sqlite-vec: {exc}") from exc
        finally:
            with contextlib.suppress(Exception):  # pragma: no cover
                conn.enable_load_extension(False)

    @property
    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    class _Tx:
        """`BEGIN IMMEDIATE` context manager.

        IMMEDIATE rather than DEFERRED: it takes the write lock up front, so
        concurrent writers queue on ``busy_timeout`` instead of racing to
        upgrade a read lock and hitting an immediate SQLITE_BUSY that no
        timeout covers.
        """

        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def __enter__(self) -> sqlite3.Connection:
            self.conn.execute("BEGIN IMMEDIATE")
            return self.conn

        def __exit__(self, exc_type: object, *_: object) -> None:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")

    def _tx(self) -> _Tx:
        return self._Tx(self.conn)

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            conn = self.conn
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

            existing = self.get_meta(META_SCHEMA_VERSION)
            if existing is None:
                self.set_meta(META_SCHEMA_VERSION, str(SCHEMA_VERSION))
            elif int(existing) > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database at {self.path} was written by schema version {existing}, "
                    f"but this build understands version {SCHEMA_VERSION}. Upgrade "
                    f"proper-search rather than downgrading the database."
                )
            elif int(existing) < SCHEMA_VERSION:
                self._migrate(int(existing))

            dim = self.get_meta(META_EMBEDDING_DIM)
            if dim:
                self._vector_dim = int(dim)
                self._create_vector_table(self._vector_dim)
            self._initialized = True

    def _migrate(self, from_version: int) -> None:
        """Apply migrations in order. No steps exist yet at version 1."""
        self.set_meta(META_SCHEMA_VERSION, str(SCHEMA_VERSION))

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- meta --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- media -------------------------------------------------------------

    def upsert_media(self, item: MediaItem) -> bool:
        with self._tx() as conn:
            existing = conn.execute("SELECT id FROM media WHERE id = ?", (item.id,)).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO media (
                        id, mime, width, height, duration_ms, frame_count, fps,
                        size_bytes, phash, thumbnail_path, status, error,
                        vision_model, vision_strategy, prompt_version,
                        created_at, indexed_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.id,
                        item.mime,
                        item.width,
                        item.height,
                        item.duration_ms,
                        item.frame_count,
                        item.fps,
                        item.size_bytes,
                        item.phash,
                        item.thumbnail_path,
                        str(item.status),
                        item.error,
                        item.vision_model,
                        item.vision_strategy,
                        item.prompt_version,
                        _iso(item.created_at),
                        _iso(item.indexed_at),
                    ),
                )
                created = True
            else:
                # Preserve created_at and never clobber a description's
                # provenance with nulls from a bare re-scan.
                conn.execute(
                    """
                    UPDATE media SET
                        mime=?, width=?, height=?, duration_ms=?, frame_count=?, fps=?,
                        size_bytes=?, phash=?,
                        thumbnail_path=COALESCE(?, thumbnail_path),
                        status=?, error=?,
                        vision_model=COALESCE(?, vision_model),
                        vision_strategy=COALESCE(?, vision_strategy),
                        prompt_version=COALESCE(?, prompt_version),
                        indexed_at=COALESCE(?, indexed_at)
                    WHERE id=?
                    """,
                    (
                        item.mime,
                        item.width,
                        item.height,
                        item.duration_ms,
                        item.frame_count,
                        item.fps,
                        item.size_bytes,
                        item.phash,
                        item.thumbnail_path,
                        str(item.status),
                        item.error,
                        item.vision_model,
                        item.vision_strategy,
                        item.prompt_version,
                        _iso(item.indexed_at),
                        item.id,
                    ),
                )
                created = False

            for source in item.sources:
                conn.execute(
                    "INSERT OR IGNORE INTO media_sources "
                    "(media_id, source_type, source_uri, first_seen_at) VALUES (?,?,?,?)",
                    (
                        item.id,
                        str(source.source_type),
                        source.source_uri,
                        _iso(source.first_seen_at),
                    ),
                )
        return created

    def _row_to_media(
        self, row: sqlite3.Row, sources: list[MediaSource] | None = None
    ) -> MediaItem:
        return MediaItem(
            id=row["id"],
            mime=row["mime"],
            width=row["width"],
            height=row["height"],
            duration_ms=row["duration_ms"],
            frame_count=row["frame_count"],
            fps=row["fps"],
            size_bytes=row["size_bytes"],
            phash=row["phash"],
            thumbnail_path=row["thumbnail_path"],
            status=MediaStatus(row["status"]),
            error=row["error"],
            vision_model=row["vision_model"],
            vision_strategy=row["vision_strategy"],
            prompt_version=row["prompt_version"],
            created_at=_parse_dt(row["created_at"]) or utcnow(),
            indexed_at=_parse_dt(row["indexed_at"]),
            sources=sources or [],
        )

    def _sources_for(self, media_ids: Sequence[str]) -> dict[str, list[MediaSource]]:
        if not media_ids:
            return {}
        placeholders = ",".join("?" * len(media_ids))
        rows = self.conn.execute(
            f"SELECT * FROM media_sources WHERE media_id IN ({placeholders})", tuple(media_ids)
        ).fetchall()
        out: dict[str, list[MediaSource]] = {}
        for row in rows:
            out.setdefault(row["media_id"], []).append(
                MediaSource(
                    source_type=SourceType(row["source_type"]),
                    source_uri=row["source_uri"],
                    first_seen_at=_parse_dt(row["first_seen_at"]) or utcnow(),
                )
            )
        return out

    def get_media(self, media_id: str) -> MediaItem | None:
        row = self.conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_media(row, self._sources_for([media_id]).get(media_id, []))

    def get_media_many(self, media_ids: Sequence[str]) -> dict[str, MediaItem]:
        ids = list(dict.fromkeys(media_ids))
        if not ids:
            return {}
        result: dict[str, MediaItem] = {}
        sources = self._sources_for(ids)
        # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER on older builds.
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT * FROM media WHERE id IN ({placeholders})", tuple(batch)
            ).fetchall()
            for row in rows:
                result[row["id"]] = self._row_to_media(row, sources.get(row["id"], []))
        return result

    def find_by_source_uri(self, source_uri: str) -> str | None:
        row = self.conn.execute(
            "SELECT media_id FROM media_sources WHERE source_uri = ? LIMIT 1", (source_uri,)
        ).fetchone()
        return row["media_id"] if row else None

    def add_source(self, media_id: str, source: MediaSource) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO media_sources "
                "(media_id, source_type, source_uri, first_seen_at) VALUES (?,?,?,?)",
                (media_id, str(source.source_type), source.source_uri, _iso(source.first_seen_at)),
            )

    def set_status(self, media_id: str, status: MediaStatus, *, error: str | None = None) -> None:
        indexed_at = _iso(utcnow()) if status is MediaStatus.INDEXED else None
        with self._tx() as conn:
            conn.execute(
                "UPDATE media SET status=?, error=?, indexed_at=COALESCE(?, indexed_at) WHERE id=?",
                (str(status), error, indexed_at, media_id),
            )

    def list_media(
        self, *, status: MediaStatus | None = None, limit: int = 100, offset: int = 0
    ) -> list[MediaItem]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM media WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (str(status), limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM media ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        sources = self._sources_for([r["id"] for r in rows])
        return [self._row_to_media(r, sources.get(r["id"], [])) for r in rows]

    def find_near_duplicates(self, phash: int, *, max_distance: int = 4) -> list[str]:
        """Bit-distance scan over stored perceptual hashes.

        A full scan, which is fine up to a few hundred thousand rows and is not
        on the search path — it runs once per ingest. Beyond that, a BK-tree or
        banded-hash index is the upgrade.
        """
        rows = self.conn.execute("SELECT id, phash FROM media WHERE phash != 0").fetchall()
        return [r["id"] for r in rows if bin(r["phash"] ^ phash).count("1") <= max_distance]

    def delete_media(self, media_id: str) -> bool:
        with self._tx() as conn:
            chunk_ids = [
                r["id"] for r in conn.execute("SELECT id FROM chunks WHERE media_id=?", (media_id,))
            ]
            self._delete_vectors(conn, chunk_ids)
            conn.execute("DELETE FROM media_fts WHERE media_id=?", (media_id,))
            conn.execute("DELETE FROM chunks_fts WHERE media_id=?", (media_id,))
            conn.execute("DELETE FROM jobs WHERE media_id=?", (media_id,))
            cursor = conn.execute("DELETE FROM media WHERE id=?", (media_id,))
            return cursor.rowcount > 0

    # -- descriptions ------------------------------------------------------

    def save_description(
        self,
        media_id: str,
        description: Description,
        *,
        model: str,
        strategy: str,
        prompt_version: str,
        raw_response: str | None = None,
    ) -> None:
        now = _iso(utcnow())
        tags = list(description.tags)
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO descriptions (
                    media_id, narrative, on_screen_text, tags_json,
                    frame_notes_json, confidence, raw_response, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(media_id) DO UPDATE SET
                    narrative=excluded.narrative,
                    on_screen_text=excluded.on_screen_text,
                    tags_json=excluded.tags_json,
                    frame_notes_json=excluded.frame_notes_json,
                    confidence=excluded.confidence,
                    raw_response=excluded.raw_response,
                    created_at=excluded.created_at
                """,
                (
                    media_id,
                    description.narrative,
                    description.on_screen_text,
                    json.dumps(tags),
                    json.dumps([n.model_dump() for n in description.frame_notes]),
                    str(description.confidence),
                    raw_response,
                    now,
                ),
            )

            # Replace rather than merge: a re-description that drops a tag must
            # actually drop it, or filters drift from the stored narrative.
            conn.execute("DELETE FROM tags WHERE media_id=?", (media_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO tags(media_id, tag) VALUES (?,?)",
                [(media_id, tag) for tag in tags],
            )

            conn.execute("DELETE FROM media_fts WHERE media_id=?", (media_id,))
            conn.execute(
                "INSERT INTO media_fts(media_id, narrative, on_screen_text, tags) VALUES (?,?,?,?)",
                (media_id, description.narrative, description.on_screen_text, " ".join(tags)),
            )

            conn.execute(
                "UPDATE media SET vision_model=?, vision_strategy=?, prompt_version=?, "
                "status=?, error=NULL WHERE id=?",
                (model, strategy, prompt_version, str(MediaStatus.DESCRIBED), media_id),
            )

    def get_description(self, media_id: str) -> Description | None:
        row = self.conn.execute(
            "SELECT * FROM descriptions WHERE media_id=?", (media_id,)
        ).fetchone()
        if row is None:
            return None
        return Description(
            narrative=row["narrative"],
            on_screen_text=row["on_screen_text"] or "",
            tags=json.loads(row["tags_json"]),
            frame_notes=[FrameNote(**n) for n in json.loads(row["frame_notes_json"])],
            confidence=Confidence(row["confidence"]) if row["confidence"] else Confidence.MEDIUM,
        )

    # -- chunks ------------------------------------------------------------

    def replace_chunks(self, media_id: str, chunks: Sequence[Chunk]) -> list[int]:
        with self._tx() as conn:
            old_ids = [
                r["id"] for r in conn.execute("SELECT id FROM chunks WHERE media_id=?", (media_id,))
            ]
            self._delete_vectors(conn, old_ids)
            conn.execute("DELETE FROM chunks WHERE media_id=?", (media_id,))
            conn.execute("DELETE FROM chunks_fts WHERE media_id=?", (media_id,))

            new_ids: list[int] = []
            for chunk in chunks:
                cursor = conn.execute(
                    "INSERT INTO chunks(media_id, ord, kind, text, t_start_ms, t_end_ms) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        media_id,
                        chunk.ord,
                        str(chunk.kind),
                        chunk.text,
                        chunk.t_start_ms,
                        chunk.t_end_ms,
                    ),
                )
                chunk_id = int(cursor.lastrowid or 0)
                new_ids.append(chunk_id)
                conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, media_id, text) VALUES (?,?,?)",
                    (chunk_id, media_id, chunk.text),
                )
            return new_ids

    def get_chunks(self, media_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE media_id=? ORDER BY ord", (media_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            media_id=row["media_id"],
            ord=row["ord"],
            kind=ChunkKind(row["kind"]),
            text=row["text"],
            t_start_ms=row["t_start_ms"],
            t_end_ms=row["t_end_ms"],
        )

    # -- vectors -----------------------------------------------------------

    def _create_vector_table(self, dimensions: int) -> None:
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE} USING vec0("
            f"  chunk_id INTEGER PRIMARY KEY,"
            f"  embedding FLOAT[{dimensions}]"
            f")"
        )

    def ensure_vector_space(self, *, model: str, dimensions: int) -> None:
        """Bind the index to one embedding model, or refuse to mix.

        Vectors from different models occupy different spaces. Mixing them
        produces results that look ranked and are noise, with no error anywhere
        — so the mismatch is caught here, loudly, instead of silently degrading
        every future search.
        """
        stored_model = self.get_meta(META_EMBEDDING_MODEL)
        stored_dim = self.get_meta(META_EMBEDDING_DIM)

        if stored_model is None:
            with self._tx():
                self.set_meta(META_EMBEDDING_MODEL, model)
                self.set_meta(META_EMBEDDING_DIM, str(dimensions))
            self._create_vector_table(dimensions)
            self._vector_dim = dimensions
            return

        if stored_model != model or int(stored_dim or 0) != dimensions:
            raise IndexConsistencyError(
                f"this index was built with embedding model {stored_model!r} "
                f"({stored_dim} dimensions), but the current configuration is "
                f"{model!r} ({dimensions} dimensions). Vectors from different models "
                f"are not comparable. Either restore the original embedding config, "
                f"or re-embed the corpus into a fresh index."
            )
        self._create_vector_table(dimensions)
        self._vector_dim = dimensions

    def _require_vector_table(self) -> int:
        if self._vector_dim is None:
            dim = self.get_meta(META_EMBEDDING_DIM)
            if dim is None:
                raise StorageError(
                    "no embedding space is bound to this index yet; "
                    "call ensure_vector_space() before storing or querying vectors"
                )
            self._vector_dim = int(dim)
            self._create_vector_table(self._vector_dim)
        return self._vector_dim

    @staticmethod
    def _delete_vectors(conn: sqlite3.Connection, chunk_ids: Iterable[int]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        # The vec0 table may not exist yet if nothing has been embedded.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.executemany(
                f"DELETE FROM {VECTOR_TABLE} WHERE chunk_id=?", [(cid,) for cid in ids]
            )

    def save_embeddings(self, pairs: Sequence[tuple[int, Sequence[float]]]) -> None:
        if not pairs:
            return
        import sqlite_vec

        dim = self._require_vector_table()
        for chunk_id, vector in pairs:
            if len(vector) != dim:
                raise IndexConsistencyError(
                    f"chunk {chunk_id} produced a {len(vector)}-dimension vector but the "
                    f"index expects {dim}. The embedding model likely changed."
                )
        with self._tx() as conn:
            payload = [(cid, sqlite_vec.serialize_float32(list(vec))) for cid, vec in pairs]
            conn.executemany(
                f"DELETE FROM {VECTOR_TABLE} WHERE chunk_id=?", [(cid,) for cid, _ in payload]
            )
            conn.executemany(
                f"INSERT INTO {VECTOR_TABLE}(chunk_id, embedding) VALUES (?,?)", payload
            )

    def chunks_missing_embeddings(self, *, limit: int = 500) -> list[Chunk]:
        try:
            self._require_vector_table()
        except StorageError:
            # Nothing embedded yet: every chunk is missing one.
            rows = self.conn.execute(
                "SELECT * FROM chunks ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]
        rows = self.conn.execute(
            f"SELECT c.* FROM chunks c "
            f"LEFT JOIN {VECTOR_TABLE} v ON v.chunk_id = c.id "
            f"WHERE v.chunk_id IS NULL ORDER BY c.id LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # -- filters -----------------------------------------------------------

    def _filter_sql(self, filters: SearchFilters | None) -> tuple[str, list[Any]]:
        """Build a WHERE fragment over the ``media`` table aliased as ``m``."""
        if filters is None or filters.is_empty():
            return "", []
        clauses: list[str] = []
        params: list[Any] = []

        if filters.tags:
            # AND semantics: every requested tag must be present.
            placeholders = ",".join("?" * len(filters.tags))
            clauses.append(
                f"(SELECT COUNT(DISTINCT tag) FROM tags "
                f"WHERE media_id = m.id AND tag IN ({placeholders})) = ?"
            )
            params.extend(filters.tags)
            params.append(len(filters.tags))

        if filters.has_on_screen_text is not None:
            op = "!=" if filters.has_on_screen_text else "="
            clauses.append(
                "COALESCE((SELECT on_screen_text FROM descriptions "
                f"WHERE media_id = m.id), '') {op} ''"
            )

        if filters.min_duration_ms is not None:
            clauses.append("m.duration_ms >= ?")
            params.append(filters.min_duration_ms)
        if filters.max_duration_ms is not None:
            clauses.append("m.duration_ms <= ?")
            params.append(filters.max_duration_ms)
        if filters.mime is not None:
            clauses.append("m.mime = ?")
            params.append(filters.mime)
        if filters.source_type is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM media_sources s "
                "WHERE s.media_id = m.id AND s.source_type = ?)"
            )
            params.append(str(filters.source_type))

        return (" AND ".join(clauses), params)

    def filter_media_ids(self, filters: SearchFilters, *, limit: int = 10_000) -> set[str] | None:
        where, params = self._filter_sql(filters)
        if not where:
            return None
        rows = self.conn.execute(
            f"SELECT m.id FROM media m WHERE {where} LIMIT ?", (*params, limit)
        ).fetchall()
        return {r["id"] for r in rows}

    # -- retrieval ---------------------------------------------------------

    def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
        weights: tuple[float, float, float] = (1.0, 3.0, 1.5),
    ) -> list[LexicalMatch]:
        match = build_fts_query(query)
        if match is None:
            return []
        where, params = self._filter_sql(filters)
        filter_clause = f" AND {where}" if where else ""

        # bm25() weights are positional and include UNINDEXED columns, so the
        # leading 0.0 corresponds to media_id. It returns negative values where
        # more negative is more relevant; negate so larger is better everywhere.
        sql = f"""
            SELECT f.media_id AS media_id,
                   -bm25(media_fts, 0.0, ?, ?, ?) AS score,
                   snippet(media_fts, 1, '', '', ' … ', 24) AS snip
            FROM media_fts f
            JOIN media m ON m.id = f.media_id
            WHERE media_fts MATCH ?{filter_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, (*weights, match, *params, limit)).fetchall()
        return [
            LexicalMatch(media_id=r["media_id"], score=float(r["score"]), snippet=r["snip"] or "")
            for r in rows
        ]

    def search_chunks_lexical(
        self, query: str, *, limit: int, filters: SearchFilters | None = None
    ) -> list[LexicalMatch]:
        match = build_fts_query(query)
        if match is None:
            return []
        where, params = self._filter_sql(filters)
        filter_clause = f" AND {where}" if where else ""
        sql = f"""
            SELECT f.media_id AS media_id, f.chunk_id AS chunk_id,
                   -bm25(chunks_fts, 0.0, 0.0, 1.0) AS score,
                   snippet(chunks_fts, 2, '', '', ' … ', 24) AS snip
            FROM chunks_fts f
            JOIN media m ON m.id = f.media_id
            WHERE chunks_fts MATCH ?{filter_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, (match, *params, limit)).fetchall()
        return [
            LexicalMatch(
                media_id=r["media_id"],
                chunk_id=int(r["chunk_id"]),
                score=float(r["score"]),
                snippet=r["snip"] or "",
            )
            for r in rows
        ]

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[DenseMatch]:
        """KNN over chunk vectors.

        Filters are applied after the KNN rather than inside it, because vec0's
        MATCH cannot take an arbitrary correlated predicate. To keep recall
        under a selective filter, the KNN over-fetches and the surplus is
        discarded — costly only in the filtered case, and correct in both.
        """
        import sqlite_vec

        dim = self._require_vector_table()
        if len(vector) != dim:
            raise IndexConsistencyError(
                f"query vector has {len(vector)} dimensions but the index expects {dim}; "
                f"the query is being embedded with a different model than the corpus"
            )

        where, params = self._filter_sql(filters)
        k = limit * 5 if where else limit
        filter_clause = f" AND {where}" if where else ""

        sql = f"""
            SELECT v.chunk_id AS chunk_id, v.distance AS distance, c.media_id AS media_id
            FROM (
                SELECT chunk_id, distance
                FROM {VECTOR_TABLE}
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            ) v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN media m ON m.id = c.media_id
            WHERE 1=1{filter_clause}
            ORDER BY v.distance
            LIMIT ?
        """
        rows = self.conn.execute(
            sql, (sqlite_vec.serialize_float32(list(vector)), k, *params, limit)
        ).fetchall()
        return [
            DenseMatch(
                media_id=r["media_id"], chunk_id=int(r["chunk_id"]), distance=float(r["distance"])
            )
            for r in rows
        ]

    # -- jobs --------------------------------------------------------------

    def enqueue(self, media_id: str, kind: JobKind) -> int:
        now = _iso(utcnow())
        with self._tx() as conn:
            # Re-queueing an item that already failed should retry it, so a
            # conflict resets state rather than being ignored.
            conn.execute(
                """
                INSERT INTO jobs
                    (media_id, kind, state, attempts, created_at, updated_at, run_after)
                VALUES (?,?,?,0,?,?,?)
                ON CONFLICT(media_id, kind) DO UPDATE SET
                    state = CASE WHEN jobs.state = 'done' THEN jobs.state ELSE 'pending' END,
                    lease_until = NULL,
                    run_after = excluded.run_after,
                    updated_at = excluded.updated_at
                """,
                (media_id, str(kind), str(JobState.PENDING), now, now, now),
            )
            row = conn.execute(
                "SELECT id FROM jobs WHERE media_id=? AND kind=?", (media_id, str(kind))
            ).fetchone()
            return int(row["id"])

    def claim_jobs(self, kind: JobKind, *, limit: int, lease_seconds: int) -> list[Job]:
        """Select and lease runnable jobs in one write transaction.

        Selecting then updating inside a single IMMEDIATE transaction is what
        makes this safe for multiple concurrent workers: two workers cannot
        observe the same pending row and both claim it.
        """
        now = utcnow()
        now_iso = _iso(now)
        lease_iso = _iso(now + timedelta(seconds=lease_seconds))
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE kind = ?
                  AND state = 'pending'
                  AND (run_after IS NULL OR run_after <= ?)
                ORDER BY id
                LIMIT ?
                """,
                (str(kind), now_iso, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [int(r["id"]) for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE jobs SET state='running', lease_until=?, attempts=attempts+1, "
                f"updated_at=? WHERE id IN ({placeholders})",
                (lease_iso, now_iso, *ids),
            )
        return [
            Job(
                id=int(r["id"]),
                media_id=r["media_id"],
                kind=JobKind(r["kind"]),
                state=JobState.RUNNING,
                attempts=int(r["attempts"]) + 1,
                last_error=r["last_error"],
                lease_until=_parse_dt(lease_iso),
                created_at=_parse_dt(r["created_at"]) or utcnow(),
                updated_at=now,
            )
            for r in rows
        ]

    def complete_job(self, job_id: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE jobs SET state='done', lease_until=NULL, last_error=NULL, updated_at=? "
                "WHERE id=?",
                (_iso(utcnow()), job_id),
            )

    def fail_job(
        self, job_id: int, error: str, *, retry: bool, backoff_seconds: float = 0.0
    ) -> None:
        now = utcnow()
        state = JobState.PENDING if retry else JobState.FAILED
        run_after = _iso(now + timedelta(seconds=backoff_seconds)) if retry else None
        with self._tx() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, lease_until=NULL, last_error=?, run_after=?, "
                "updated_at=? WHERE id=?",
                (str(state), error[:2000], run_after, _iso(now), job_id),
            )

    def reclaim_expired_leases(self) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET state='pending', lease_until=NULL, updated_at=? "
                "WHERE state='running' AND lease_until IS NOT NULL AND lease_until < ?",
                (_iso(utcnow()), _iso(utcnow())),
            )
            return cursor.rowcount

    def list_jobs(
        self, *, state: str | None = None, kind: JobKind | None = None, limit: int = 100
    ) -> list[Job]:
        clauses, params = [], []
        if state:
            clauses.append("state=?")
            params.append(state)
        if kind:
            clauses.append("kind=?")
            params.append(str(kind))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        ).fetchall()
        return [
            Job(
                id=int(r["id"]),
                media_id=r["media_id"],
                kind=JobKind(r["kind"]),
                state=JobState(r["state"]),
                attempts=int(r["attempts"]),
                last_error=r["last_error"],
                lease_until=_parse_dt(r["lease_until"]),
                created_at=_parse_dt(r["created_at"]) or utcnow(),
                updated_at=_parse_dt(r["updated_at"]) or utcnow(),
            )
            for r in rows
        ]

    # -- introspection -----------------------------------------------------

    def stats(self) -> IndexStats:
        conn = self.conn
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) n FROM media GROUP BY status")
        }
        by_job_state = {
            r["state"]: r["n"]
            for r in conn.execute("SELECT state, COUNT(*) n FROM jobs GROUP BY state")
        }
        vector_total = 0
        try:
            self._require_vector_table()
            vector_total = int(
                conn.execute(f"SELECT COUNT(*) n FROM {VECTOR_TABLE}").fetchone()["n"]
            )
        except (StorageError, sqlite3.OperationalError):
            pass

        dim = self.get_meta(META_EMBEDDING_DIM)
        return IndexStats(
            media_total=int(conn.execute("SELECT COUNT(*) n FROM media").fetchone()["n"]),
            media_by_status=by_status,
            chunk_total=int(conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]),
            vector_total=vector_total,
            tag_total=int(conn.execute("SELECT COUNT(DISTINCT tag) n FROM tags").fetchone()["n"]),
            jobs_by_state=by_job_state,
            embedding_model=self.get_meta(META_EMBEDDING_MODEL),
            embedding_dim=int(dim) if dim else None,
        )
