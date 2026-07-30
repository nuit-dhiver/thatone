"""Splitting a description into independently embedded chunks.

This is the design decision the whole retrieval quality rests on.

One vector per clip is the obvious approach and it fails at exactly the queries
this system exists to serve. Averaging "a man sits at a desk, then stands, then
knocks over a lamp, then walks out" into a single vector produces something
close to every clip about a man in an office and close to nothing in
particular. Searching for "the bit where he knocks the lamp over" then competes
against the whole summary rather than matching the moment.

Chunking per frame note fixes that: the moment gets its own vector, its own
timestamp, and its own shot at ranking. A clip scores by its *best* chunk, so a
strong match on one instant beats a diffuse match across a summary.
"""

from __future__ import annotations

import re

from ..models import Chunk, ChunkKind, Description

# Split on sentence-ending punctuation followed by whitespace. Deliberately
# simple: descriptions are model-generated prose, not arbitrary text with
# abbreviations and decimals to trip over.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

MIN_CHUNK_CHARS = 24
"""Below this a chunk carries no retrievable meaning and just adds a vector
that matches everything weakly. Short fragments get folded into a neighbour."""


def split_text(text: str, *, max_chars: int) -> list[str]:
    """Split prose into pieces no longer than ``max_chars``, on sentence bounds.

    Sentences are kept whole where possible: a vector for half a sentence
    embeds a fragment whose meaning depends on the missing half.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        # A single sentence longer than the limit has no good split point;
        # hard-wrap it rather than emitting an over-long chunk.
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars].strip())
            sentence = sentence[max_chars:]
        current = sentence.strip()
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def build_chunks(
    media_id: str, description: Description, *, max_chars: int = 1200
) -> list[Chunk]:
    """Turn one description into the chunks that get embedded.

    Three kinds, each earning its place:

    * ``narrative`` — the whole-clip story, for topical queries ("cat knocking
      things off a table").
    * ``frame`` — one per observed moment, timestamped, for the moment queries
      that motivate the whole design.
    * ``screen_text`` — burned-in captions on their own, so a remembered quote
      matches a chunk that is *only* that quote instead of one sentence buried
      in a paragraph.
    """
    chunks: list[Chunk] = []
    ordinal = 0

    for piece in split_text(description.narrative, max_chars=max_chars):
        chunks.append(
            Chunk(media_id=media_id, ord=ordinal, kind=ChunkKind.NARRATIVE, text=piece)
        )
        ordinal += 1

    if description.on_screen_text:
        for piece in split_text(description.on_screen_text, max_chars=max_chars):
            chunks.append(
                Chunk(media_id=media_id, ord=ordinal, kind=ChunkKind.SCREEN_TEXT, text=piece)
            )
            ordinal += 1

    # Frame notes are written to stand alone, but a bare "He stands up." embeds
    # almost identically for every clip containing a person. Prefixing the tags
    # gives the vector enough subject context to be discriminative while
    # keeping the moment itself as the head of the text.
    context = ", ".join(description.tags[:6])
    for note in description.frame_notes:
        text = note.note.strip()
        if len(text) < MIN_CHUNK_CHARS and not context:
            continue
        enriched = f"{text} ({context})" if context else text
        chunks.append(
            Chunk(
                media_id=media_id,
                ord=ordinal,
                kind=ChunkKind.FRAME,
                text=enriched,
                t_start_ms=note.t_ms,
                t_end_ms=note.t_ms,
            )
        )
        ordinal += 1

    return chunks


def estimate_tokens(chunks: list[Chunk]) -> int:
    """Rough embedding-token count for a chunk set (~4 characters per token)."""
    return sum(max(1, len(c.text) // 4) for c in chunks)
