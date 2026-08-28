"""Shared parsing helpers.

Everything here is about the gap between what a transcript format promises and
what real files actually contain. Each helper exists because some vendor's
export broke a naive implementation.
"""

from __future__ import annotations

import html
import re
from typing import List, Optional

from ..types import Turn

# HH:MM:SS.mmm | HH:MM:SS,mmm | MM:SS.mmm | HH:MM:SS
_TS = re.compile(
    r"(?:(?P<h>\d{1,3}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?"
)

# WebVTT/SRT inline markup: <v Alice>, </v>, <c.colorE5E5E5>, <i>, <b>, <u>,
# and YouTube's per-word timestamps <00:00:01.234>.
_TAG = re.compile(r"</?[a-zA-Z][^>]*>|<\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}>")

# Caption formatting tags only, applied after entity decoding. A closed list,
# so ordinary angle-bracket text in a transcript is never eaten.
_ESCAPED_TAG = re.compile(
    r"</?(?:i|b|u|c|v|em|strong|ruby|rt|lang|font|br)\b[^>]*/?>", re.IGNORECASE
)

# <v Alice> and <v.loud Alice> -- the Teams/WebVTT voice span.
_VOICE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+?)\s*>", re.IGNORECASE)

# "Alice:" / "ALICE JONES:" / "Dr. Smith:" at the start of a line.
# Deliberately strict: at most six words, no sentence-ending punctuation, and a
# length cap -- otherwise "So here's the thing: we need to decide" is read as a
# speaker named "So here's the thing".
_INLINE_SPEAKER = re.compile(
    r"^\s*(?P<name>[^\s:][^:\n]{0,48}?)\s*:\s+(?P<rest>\S.*)$", re.DOTALL
)
# Prefixes that look like a speaker label but aren't.
_NOT_A_SPEAKER = {
    "note", "notes", "warning", "caution", "example", "tip", "update", "edit",
    "source", "sources", "ref", "reference", "todo", "fixme", "summary",
    "action", "actions", "e.g", "i.e", "http", "https", "subject", "re",
    "from", "to", "cc", "date", "time", "title", "topic", "agenda",
}
_BAD_IN_NAME = re.compile(r"[,;!?]|\bhttps?\b", re.IGNORECASE)


def parse_timestamp(text: str) -> Optional[int]:
    """Parse a timestamp to milliseconds, or None if it isn't one."""
    m = _TS.fullmatch(text.strip())
    if not m:
        return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m"))
    s = int(m.group("s"))
    frac = m.group("ms") or "0"
    ms = int(frac.ljust(3, "0")[:3])
    return ((h * 60 + mi) * 60 + s) * 1000 + ms


def strip_tags(text: str) -> str:
    """Remove caption markup and decode HTML entities.

    Two passes, because producers disagree about escaping. The WebVTT spec says
    ``&lt;i&gt;`` is the literal text ``<i>``, but YouTube emits exactly that to
    mean italics -- so stripping only raw tags leaves ``<i>`` sitting in the
    output text. Unescaping first is not safe either: it would turn genuinely
    escaped content into something the tag regex eats.

    So: strip real tags, unescape, then strip a second time using a *closed
    list* of caption formatting tags. A run of ``<div>`` or ``a < b`` in the
    transcript survives; ``<i>`` and ``<c.colorE5E5E5>`` do not.
    """
    cleaned = html.unescape(_TAG.sub("", text))
    return _ESCAPED_TAG.sub("", cleaned)


def extract_voice_speaker(text: str) -> Optional[str]:
    """Pull the speaker out of a ``<v Alice>`` voice span, if present."""
    m = _VOICE.search(text)
    return m.group(1).strip() if m else None


def split_inline_speaker(text: str) -> tuple[Optional[str], str]:
    """Split ``"Alice: hello"`` into ``("Alice", "hello")``.

    Returns ``(None, text)`` when the prefix does not look like a name, which is
    the common case for ordinary sentences containing a colon.
    """
    m = _INLINE_SPEAKER.match(text)
    if not m:
        return None, text
    name = m.group("name").strip()
    if not _looks_like_speaker(name):
        return None, text
    return name, m.group("rest").strip()


def _looks_like_speaker(name: str) -> bool:
    """Is this colon-prefix a speaker label, or just a sentence with a colon?

    Getting this wrong in either direction is expensive: miss a real label and
    every turn becomes anonymous; accept a false one and "So here's the thing:"
    becomes a speaker who says one line and never appears again. The rule is
    conservative -- a label is a single word, or several words that all start
    capitalised, and never a phrase containing sentence punctuation.
    """
    if not name or len(name) > 48:
        return False
    words = name.split()
    if not words or len(words) > 6:
        return False
    if _BAD_IN_NAME.search(name) or name.endswith("."):
        return False
    if not any(c.isalnum() for c in name):
        return False
    if name.lower().strip(".") in _NOT_A_SPEAKER:
        return False
    if len(words) == 1:
        return True
    # Multi-word labels must read as a name: every word capitalised (or a
    # digit/underscore token like "SPEAKER 00").
    for w in words:
        first = w[0]
        if not (first.isupper() or first.isdigit() or first == "_"):
            return False
    return True


def drop_rolling_duplicates(prev_lines: List[str], lines: List[str]) -> List[str]:
    """Remove the repeated prefix produced by rolling captions.

    YouTube auto-captions scroll: each cue repeats the tail of the previous cue
    and appends one new line. A parser that concatenates cues naively produces a
    transcript where almost every sentence appears two or three times, which
    quietly doubles the size of the index and wrecks retrieval. This finds the
    longest overlap between the end of what we already have and the start of the
    new cue, and drops it.
    """
    if not prev_lines or not lines:
        return lines
    max_k = min(len(prev_lines), len(lines))
    for k in range(max_k, 0, -1):
        if prev_lines[-k:] == lines[:k]:
            return lines[k:]
    return lines


def merge_turns(
    cues: List[Turn],
    *,
    max_gap_ms: Optional[int] = None,
    join: str = " ",
) -> List[Turn]:
    """Merge consecutive cues by the same speaker into single turns.

    Subtitle formats emit a cue every few seconds; a *turn* is everything one
    person said before someone else spoke. Chunking on cues instead of turns is
    the mistake that makes speaker-aware chunking pointless, because a cue
    boundary falls wherever the caption renderer needed a line break.

    ``max_gap_ms`` optionally starts a new turn when the same speaker resumes
    after a long silence, which is usually a genuine topic break.
    """
    out: List[Turn] = []
    # Text is accumulated per turn in a list and joined once at the end.
    # Appending to `turn.text` directly rebuilds the whole string on every cue,
    # which is quadratic: a 16 MB auto-caption file whose 46,958 cues all belong
    # to one speaker spent 1.3 seconds doing nothing but copying characters.
    parts: List[List[str]] = []

    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        if out and out[-1].speaker == cue.speaker:
            gap_ok = True
            if max_gap_ms is not None and out[-1].end_ms is not None and cue.start_ms is not None:
                gap_ok = (cue.start_ms - out[-1].end_ms) <= max_gap_ms
            if gap_ok:
                prev = out[-1]
                parts[-1].append(text)
                if cue.end_ms is not None:
                    prev.end_ms = cue.end_ms
                if prev.start_ms is None and cue.start_ms is not None:
                    prev.start_ms = cue.start_ms
                continue
        out.append(
            Turn(
                text="",
                speaker=cue.speaker,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                index=len(out),
                raw_speaker=cue.raw_speaker,
                meta=dict(cue.meta),
            )
        )
        parts.append([text])

    for i, (turn, chunks) in enumerate(zip(out, parts)):
        turn.text = join.join(chunks) if len(chunks) > 1 else chunks[0]
        turn.index = i
    return out
