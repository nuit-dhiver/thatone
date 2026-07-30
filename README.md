# proper-search

Find GIFs and videos by **what happens inside them**, not by their filename.

A vision model watches sampled frames, writes down what happens, and that
writing is indexed for hybrid keyword + semantic retrieval. So the query is
whatever you actually remember:

```
"the guy who slowly turns around looking completely done"
"cat knocking a glass off a table"
"the one where it says NOPE"
```

Python library + HTTP API. GIF-first, video-ready.

> **This package never runs a model.** Every provider is an HTTP client. To use
> a self-hosted model, run it on your own server and point `vision.base_url` at
> it — from here it is indistinguishable from any other API.

---

## Install

```bash
pip install -e ".[anthropic,api]"
```

Requires Python 3.11+. FFmpeg comes bundled with the `av` wheel; no separate
install needed.

---

## Quick start

```python
import asyncio
from proper_search import ProperSearch, Settings

async def main():
    async with ProperSearch.open(Settings()) as engine:
        await engine.index(["./gifs"])

        hits, diagnostics = await engine.search("cat knocking a glass off a table")
        for hit in hits:
            when = f" @{hit.snippet_t_ms}ms" if hit.snippet_t_ms is not None else ""
            print(f"{hit.score:.4f}  {hit.media.id[:12]}{when}  {hit.snippet}")

asyncio.run(main())
```

Set the keys first:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

### Check the cost before a large run

Indexing 100k clips is a real bill. Get a number first — it samples actual
items and measures them rather than multiplying a nominal frame count:

```python
estimate = await engine.estimate(["./gifs"], sample_size=10)
print(estimate.summary())
# 100000 items x $0.00558 = $558.31 (claude-sonnet-5, single_call, measured, batch)
```

### HTTP API

```bash
uvicorn proper_search.api:create_app --factory --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /index` | Ingest paths, directories, or URLs |
| `GET /search?q=` | Search; supports `tags`, `has_text`, duration filters, `explain` |
| `POST /estimate` | Project the cost of a run before committing |
| `POST /drain` | Run queued work to completion |
| `GET /media/{id}` | Item plus its description |
| `GET /media/{id}/thumbnail` | Poster frame, or `?animated=true` |
| `POST /media/{id}/reindex` | Re-describe, or `?redescribe=false` to only re-embed |
| `GET /jobs?state=failed` | What failed and why |
| `GET /stats`, `GET /healthz` | Index counts; liveness and capabilities |

Interactive docs at `/docs`.

---

## How it works

```
ingest ──► sample ──► describe ──► chunk ──► embed ──► search
 hash      adaptive    vision      per-      vectors    BM25 + dense
 probe     frames      model       moment               fused by RRF
```

**Identity is the content hash.** The same GIF from a local path and a URL is
one item with two sources — described, and paid for, once.

**Sampling adapts to content.** Most GIFs run 1–4 seconds, so "a frame every 2
seconds" gets you one frame and the model never sees the thing that happens.
Perceptual-hash scene detection sizes the sample to the clip instead: a 12-frame
reaction GIF yields ~3 distinct moments, a 60-second mostly-static clip yields
its actual beats rather than 30 near-identical frames you pay for.

**Descriptions are written for retrieval, not captioning.** The model produces a
narrative, per-frame notes, tags, and a dedicated verbatim `on_screen_text`
field — because "I remember it said NOPE" is one of the most common ways people
recall a clip.

**Chunking is per moment.** One vector per clip averages "sits, stands, knocks
over a lamp, leaves" into something close to every office clip and close to
nothing in particular. Each frame note gets its own embedding and timestamp, and
a clip ranks on its *best* chunk — so a query about one instant matches that
instant.

**Retrieval fuses two signals.** BM25 catches exact words a vector would miss (a
name, a quoted caption); vectors catch paraphrase, which is most of what people
type. Reciprocal rank fusion combines them without needing their scores to be
comparable. If one signal fails, search degrades to the other and *says so* in
`degraded_signals` rather than quietly returning worse results.

---

## Configuration

Layered, highest priority first: keyword arguments → environment →
`proper-search.yaml` → defaults. Nested values use a double underscore:

```bash
export PROPER_SEARCH__VISION__MODEL=claude-opus-5
```

See [`proper-search.example.yaml`](proper-search.example.yaml) for every option
annotated. Credentials are named by environment variable (`api_key_env`), never
inlined, so a config file is safe to commit.

### The dials that matter

| Setting | Effect |
|---|---|
| `sampling.max_frames` | Main cost lever — request cost is ~linear in frames |
| `sampling.frame_max_edge` | The other one; image tokens scale with area |
| `vision.strategy` | `single_call` (default) / `sequential` / `two_pass` |
| `vision.use_batch_api` | Half price, results within the hour. Right for backfills |
| `search.weight_on_screen_text` | How much a remembered quote outranks prose |

### Description strategies

| Strategy | Requests/item | When |
|---|---|---|
| `single_call` | 1 | **Default.** All frames in one message, so the model sees the whole sequence before writing — cross-frame revision for one request |
| `sequential` | N | One per frame, each revising the running story. Highest fidelity, ~N× the cost |
| `two_pass` | N+1 | Cheap parallel captions, then one strong-model synthesis |

---

## Measuring quality

There is no UI, so relevance regressions are invisible unless measured. The eval
harness is the feedback loop:

```python
from proper_search.eval import EvalHarness, GoldenQuery, format_report

golden = [
    GoldenQuery(query="the guy who slowly turns around",
                expected=["<media_id>"],
                note="vague, no literal keyword overlap"),
]
results = await EvalHarness(engine.store, engine.embedder, engine.settings.search).run(golden)
print(format_report(results))
```

```
search quality over 13 queries

lexical only           r@1=0.92  r@5=0.92  r@10=0.92  mrr=0.923  misses=1
dense only             r@1=0.92  r@5=0.92  r@10=0.92  mrr=0.923  misses=1
fused                  r@1=0.92  r@5=0.92  r@10=0.92  mrr=0.923  misses=1

queries 'fused' could not answer:
  - 'a purple hippopotamus doing taxes'  (not in the corpus)
```

Each signal is scored in isolation as well as fused, because the question worth
answering is not "is search good" but "is fusion earning its keep". Write the
queries the way people actually recall clips — vague, partial, occasionally
wrong. A golden set of clean keyword queries measures nothing this system was
built for.

---

## Running at scale

Indexing is a long batch job that *will* fail partway through. It is built to
survive that:

- Stages (`fetch` → `describe` → `embed`) are separate jobs, so an embedding
  outage retries only embedding and the vision spend already banked stays banked.
- Claims are leased. Kill a worker and its in-flight work returns to the pool
  rather than stranding.
- Re-running skips completed work. Re-scanning an indexed corpus costs nothing.

```python
await engine.index(["./gifs"], drain=False)   # register the work
await engine.drain()                           # or run a worker elsewhere
```

Retries cover 429/5xx/timeouts with jittered backoff. Refusals, decode failures,
and 400s are terminal — recorded with a reason, never retried — and surface at
`GET /jobs?state=failed`. Bad credentials stop the worker instead of marching the
whole queue into a failed state.

---

## Development

```bash
uv venv && uv pip install -e ".[dev,anthropic,api]"
pytest && ruff check src tests && mypy
```

The suite runs the full pipeline against deterministic stub providers — no API
key, no network, no cost — so ingest, dedupe, ranking, resumability, and error
handling are all covered in CI.

Tests that hit a real provider are marked `live` and excluded by default:

```bash
pytest -m live      # costs money
```

---

## Status

Working: storage, media, vision (Anthropic), embeddings (OpenAI/Voyage/Cohere),
hybrid search, jobs, HTTP API, cost estimation, eval harness.

Not built yet: the Postgres backend, the Gemini and OpenAI-compatible vision
adapters, and the Celery queue adapter — all have interfaces in place and raise
a clear error if selected.
