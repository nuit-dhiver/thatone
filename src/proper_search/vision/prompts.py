"""Prompts for the description pass.

This is the highest-leverage text in the system: it decides what ends up in the
index, and therefore what can ever be found. Two things shape it.

**It is written for retrieval, not for captioning.** A caption model says "a
man at a desk". Someone hunting for that GIF six months later types "the guy
who slowly turns around looking done with everything". The prompt pushes for
the second register: the beat that makes the clip worth sending, the emotional
read, the specific visual details people actually retain.

**Versioning is load-bearing.** ``PROMPT_VERSION`` is stamped on every stored
description. Bump it on any change that alters output shape or emphasis, or
rows written before and after become quietly incomparable and re-index logic
cannot tell which need redoing.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import FrameSample

PROMPT_VERSION = "v1"


# The system prompt is identical on every call, which makes it the natural
# prompt-cache prefix. It is also deliberately long: caching only engages above
# a per-model minimum prefix (512 tokens on Opus 5, 1024 on Sonnet 5, 4096 on
# Haiku 4.5), and a prompt below that line silently never caches. Verify with
# usage.cache_read_input_tokens rather than assuming.
SYSTEM = """\
You are building a searchable index of short video clips and GIFs. For each \
clip you receive frames sampled in time order, each labelled with its timestamp.

Your descriptions are the only thing that makes these clips findable. People \
will search for them months later using fragments of memory — half-remembered \
actions, a quoted caption, the feeling of the thing — so write for that reader, \
not for a caption benchmark.

## What to produce

**frame_notes** — one note per frame you were shown, in time order, tagged with \
that frame's timestamp. Describe what is visible and what is happening at that \
instant. Note changes from the previous frame explicitly: a new subject, a \
changed expression, a camera move, a cut. These notes are indexed individually, \
so each should stand on its own without the others for context.

**narrative** — what happens across the clip, as a short account in time order. \
This is not a summary of your notes; it is the story they add up to. Later \
frames routinely reveal that an earlier frame was misread — a person who looked \
angry was laughing, an object that looked like a phone was a remote. When that \
happens, write the narrative according to what you now know and do not preserve \
the earlier mistake. State what the clip is, then what happens in it.

**on_screen_text** — every piece of text visible anywhere in the frames, \
transcribed exactly as written: captions, subtitles, signs, watermarks, UI text, \
labels. Preserve original casing and spelling, including deliberate misspellings \
and ALL CAPS. Separate distinct pieces with " | ". This field carries \
disproportionate weight when someone remembers a quote, so accuracy matters more \
here than anywhere else. Use an empty string if there is genuinely no text.

**tags** — short lowercase keywords covering: the subjects (people, animals, \
objects), the actions, the setting, the emotional register, the visual style, \
and anything distinctive. Prefer the words a person would actually search for. \
Include the obvious ones; do not hunt for clever ones.

**confidence** — "high" when the frames clearly show what is happening, \
"medium" when you are inferring across gaps, "low" when the frames are too \
few, too dark, or too ambiguous to be sure.

## How to write

Describe what is actually visible. Do not invent dialogue, backstory, or a \
source you were not shown — if a clip looks like it is from a film or show and \
you genuinely recognize it, you may name it, but do not guess.

Write plainly and concretely. Skip hedging ("appears to be", "seems to", \
"possibly") — it adds nothing and dilutes the text that gets indexed. If you \
are unsure, lower the confidence field instead of hedging in the prose.

Name what is emotionally legible: someone is unimpressed, delighted, resigned, \
smug. This is often the whole reason the clip exists and the main thing people \
remember about it.

Do not describe the frames as frames in the narrative. "The first frame shows a \
man" is wrong; "a man sits at a desk" is right. The narrative is about the clip, \
not about your inputs.

Be specific about the things people use to search. Colours, clothing, breeds, \
locations, and objects are recall handles. "A person with an animal" is nearly \
useless; "a man in a yellow raincoat holding a wet golden retriever" is findable.\
"""


def _frame_manifest(frames: Sequence[FrameSample]) -> str:
    """Timestamps for the frames, so the model can key its notes to them."""
    lines = [
        f"  {i + 1}. t={frame.t_ms}ms" for i, frame in enumerate(frames)
    ]
    return "\n".join(lines)


def single_call_instruction(frames: Sequence[FrameSample], *, duration_ms: int) -> str:
    """Instruction for the default one-request strategy.

    All frames arrive in one message, so the model sees the whole sequence
    before writing anything. That is where the "revise the story as later
    frames reveal more" behaviour comes from — not from a separate revision
    pass, but from never having committed to a reading in the first place.
    """
    count = len(frames)
    plural = "frame" if count == 1 else "frames"
    return (
        f"Here are {count} {plural} sampled from a clip lasting {duration_ms}ms, "
        f"in time order:\n{_frame_manifest(frames)}\n\n"
        f"Study all {count} {plural} before writing anything. Then produce the "
        f"frame notes, the narrative, the on-screen text, the tags, and your "
        f"confidence."
    )


def sequential_first_instruction(frame: FrameSample, *, duration_ms: int, total: int) -> str:
    """Opening instruction for the frame-at-a-time strategy."""
    return (
        f"This is frame 1 of {total} from a clip lasting {duration_ms}ms, at "
        f"t={frame.t_ms}ms.\n\n"
        f"Describe what you can see. You have not seen the rest of the clip yet, "
        f"so keep the narrative to what this frame supports and set confidence "
        f"accordingly."
    )


def sequential_next_instruction(
    frame: FrameSample, *, position: int, total: int, running: str
) -> str:
    """Continuation instruction carrying the running description forward.

    The revision clause is the point of this strategy. Without an explicit
    licence to overwrite, models tend to append to the story they already told
    rather than correct it, and an early misreading survives to the end.
    """
    return (
        f"This is frame {position} of {total}, at t={frame.t_ms}ms.\n\n"
        f"Here is the description you wrote from the earlier frames:\n\n"
        f"{running}\n\n"
        f"Incorporate this new frame. If it shows that something in the earlier "
        f"description was wrong, rewrite that part — do not preserve a mistake "
        f"for consistency, and do not narrate the correction. Return the complete "
        f"updated description covering everything you have seen so far, including "
        f"a frame note for this frame appended to the existing ones."
    )


def caption_instruction(frame: FrameSample) -> str:
    """Per-frame pass of the two-pass strategy, run on a cheap model.

    Deliberately narrow: this pass has no temporal context, so asking it for a
    story invites invention. It reports, the synthesis pass interprets.
    """
    return (
        f"This is a single frame from a clip, at t={frame.t_ms}ms.\n\n"
        f"In two or three sentences, describe exactly what is visible: subjects, "
        f"what they are doing, the setting, and any text shown. Do not speculate "
        f"about what happens before or after — you are only reporting this frame."
    )


def synthesis_instruction(
    captions: Sequence[tuple[int, str]], *, duration_ms: int
) -> str:
    """Second pass of the two-pass strategy: merge captions into a description."""
    joined = "\n\n".join(f"t={t_ms}ms: {text}" for t_ms, text in captions)
    return (
        f"Below are independent per-frame observations from a clip lasting "
        f"{duration_ms}ms, in time order. They were written without any temporal "
        f"context, so some will misread things that the sequence makes obvious.\n\n"
        f"{joined}\n\n"
        f"Work out what actually happens across the clip and produce the full "
        f"description. Where an individual observation conflicts with what the "
        f"sequence implies, trust the sequence. Reuse these observations as the "
        f"frame notes, corrected where the sequence shows they were wrong."
    )


def rerank_system() -> str:
    """System prompt for the optional relevance reranker."""
    return (
        "You score how well a clip matches a search query. The searcher is "
        "recalling a clip they have seen before, so their query is often vague, "
        "partial, or slightly wrong in its details. Reward a clip that is "
        "plausibly the one they mean; do not require the query to match the "
        "description literally. Respond only with a number from 0 to 10."
    )


def rerank_instruction(query: str, narrative: str, on_screen_text: str) -> str:
    text_part = f'\nOn-screen text: "{on_screen_text}"' if on_screen_text else ""
    return (
        f'Search query: "{query}"\n\n'
        f"Clip description: {narrative}{text_part}\n\n"
        f"Score 0-10 for how likely this is the clip the searcher means. "
        f"Answer with the number only."
    )
