-- thatone SQLite schema, version 1.
--
-- Connection pragmas (WAL, busy_timeout, foreign_keys) are set per-connection
-- in backend.py, not here, because they are connection state rather than
-- schema. The sqlite-vec virtual table is created lazily once the embedding
-- width is known -- see ensure_vector_space().

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- --------------------------------------------------------------------------
-- Media
-- --------------------------------------------------------------------------

-- Identity is the sha256 of the raw bytes. The same GIF fetched from a local
-- path and from a URL is one row here with two rows in media_sources, so it is
-- described (and paid for) exactly once.
CREATE TABLE IF NOT EXISTS media (
    id              TEXT PRIMARY KEY,
    mime            TEXT    NOT NULL,
    width           INTEGER NOT NULL DEFAULT 0,
    height          INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    frame_count     INTEGER NOT NULL DEFAULT 0,
    fps             REAL    NOT NULL DEFAULT 0,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    phash           INTEGER NOT NULL DEFAULT 0,
    thumbnail_path  TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    error           TEXT,
    vision_model    TEXT,
    vision_strategy TEXT,
    prompt_version  TEXT,
    created_at      TEXT    NOT NULL,
    indexed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_media_status   ON media(status);
CREATE INDEX IF NOT EXISTS idx_media_phash    ON media(phash);
CREATE INDEX IF NOT EXISTS idx_media_duration ON media(duration_ms);
CREATE INDEX IF NOT EXISTS idx_media_mime     ON media(mime);

CREATE TABLE IF NOT EXISTS media_sources (
    media_id      TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    source_type   TEXT NOT NULL,
    source_uri    TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY (media_id, source_type, source_uri)
);

-- Lets a re-scan resolve a path or URL to a known item and skip the download.
CREATE INDEX IF NOT EXISTS idx_sources_uri  ON media_sources(source_uri);
CREATE INDEX IF NOT EXISTS idx_sources_type ON media_sources(source_type);

-- --------------------------------------------------------------------------
-- Descriptions
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS descriptions (
    media_id         TEXT PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    narrative        TEXT NOT NULL,
    on_screen_text   TEXT NOT NULL DEFAULT '',
    tags_json        TEXT NOT NULL DEFAULT '[]',
    frame_notes_json TEXT NOT NULL DEFAULT '[]',
    confidence       TEXT,
    -- Kept so a prompt-schema change can be reprocessed from the stored
    -- response instead of re-billing the vision model.
    raw_response     TEXT,
    created_at       TEXT NOT NULL
);

-- Normalized at write time so tag filtering is an index lookup, not a JSON scan.
CREATE TABLE IF NOT EXISTS tags (
    media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (media_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

-- --------------------------------------------------------------------------
-- Chunks
-- --------------------------------------------------------------------------

-- One embedding per chunk. Frame notes become their own chunks so a query
-- about a single moment matches that moment directly rather than competing
-- against a whole-clip summary averaged into one vector.
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id   TEXT    NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    t_start_ms INTEGER,
    t_end_ms   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_media ON chunks(media_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind  ON chunks(kind);

-- --------------------------------------------------------------------------
-- Lexical indexes
-- --------------------------------------------------------------------------

-- Not external-content tables: every write goes through this backend inside
-- one transaction, so explicit population is guaranteed consistent and avoids
-- a class of trigger-desync bugs. The duplicated text costs ~100MB at 100k
-- items, which is a fair trade.
--
-- Column order matters: bm25() weights are positional, and the UNINDEXED
-- media_id still occupies position 1.
CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    media_id UNINDEXED,
    narrative,
    on_screen_text,
    tags,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    media_id UNINDEXED,
    text,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

-- --------------------------------------------------------------------------
-- Jobs
-- --------------------------------------------------------------------------

-- The resumable work queue. One row per (item, stage): a failure in embedding
-- retries only embedding, never re-billing the vision call that preceded it.
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id    TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    state       TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    -- Claim expiry. A worker killed mid-job leaves a stale lease; once it
    -- passes, the job returns to the pool instead of stranding forever.
    lease_until TEXT,
    -- Earliest retry time, for exponential backoff after a transient failure.
    run_after   TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    UNIQUE (media_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(kind, state, run_after);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(state, lease_until);
CREATE INDEX IF NOT EXISTS idx_jobs_media ON jobs(media_id);
