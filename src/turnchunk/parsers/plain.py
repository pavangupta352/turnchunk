"""Plain-text transcript parser.

"Plain text" is not one format. It is every convention anyone ever invented for
writing down who said what, and real archives mix several in one folder::

    [00:12:34] Alice: text          # bracketed timestamp
    00:12:34 Alice: text            # bare timestamp
    (12:34) Alice: text             # parenthesised
    Alice: text                     # no timestamp at all
    **Alice:** text                 # markdown export
    Alice Smith  0:12               # Otter: name and time on their own line
    text on the following line

Lines carrying neither a speaker nor a timestamp continue the previous turn,
which is how wrapped paragraphs survive.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..types import Transcript, Turn
from .base import merge_turns, parse_timestamp, split_inline_speaker

_LEADING_TS = re.compile(
    r"^\s*[\[\(<]?\s*(?P<ts>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s*[\]\)>]?\s*[-–—]?\s*(?P<rest>.*)$"
)
# Otter/Descript: a line that is only "Name  12:34"
_NAME_THEN_TS = re.compile(
    r"^\s*(?P<name>[^\d:][^\n]{0,48}?)\s{1,}[\[\(]?(?P<ts>\d{1,3}:\d{2}(?::\d{2})?)[\]\)]?\s*$"
)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_SPEAKER_WITH_TS = re.compile(
    r"^\s*(?P<name>[^:\n]{1,48}?)\s*[\[\(]\s*(?P<ts>\d{1,3}:\d{2}(?::\d{2})?)\s*[\]\)]\s*:\s*(?P<rest>.*)$"
)


def looks_like_plain(text: str) -> bool:
    sample = [ln for ln in text.split("\n")[:200] if ln.strip()]
    if not sample:
        return False
    hits = 0
    for line in sample:
        stripped = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), line)
        if _LEADING_TS.match(stripped) or _NAME_THEN_TS.match(stripped):
            hits += 1
            continue
        if split_inline_speaker(stripped)[0] is not None:
            hits += 1
    return hits >= max(2, len(sample) * 0.15)


def parse_plain(
    text: str,
    *,
    source: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
) -> Transcript:
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: List[Turn] = []
    pending_speaker: Optional[str] = None
    pending_ts: Optional[int] = None

    for raw_line in lines:
        line = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), raw_line).strip()
        if not line:
            continue

        # "Alice Smith  0:12" on its own line -- the next lines are their words.
        m = _NAME_THEN_TS.match(line)
        if m and split_inline_speaker(m.group("name") + ": x")[0] is not None:
            pending_speaker = m.group("name").strip()
            pending_ts = parse_timestamp(m.group("ts"))
            continue

        # "Alice (00:12): text"
        m = _SPEAKER_WITH_TS.match(line)
        if m and split_inline_speaker(m.group("name") + ": x")[0] is not None:
            _emit(cues, m.group("rest"), m.group("name").strip(), parse_timestamp(m.group("ts")))
            pending_speaker = m.group("name").strip()
            pending_ts = None
            continue

        ts: Optional[int] = None
        m = _LEADING_TS.match(line)
        if m and m.group("rest"):
            ts = parse_timestamp(m.group("ts"))
            line = m.group("rest").strip()

        speaker, rest = split_inline_speaker(line)
        if speaker is not None:
            _emit(cues, rest, speaker, ts if ts is not None else pending_ts)
            pending_speaker = speaker
            pending_ts = None
            continue

        # No speaker on this line: it continues the current one.
        _emit(cues, line, pending_speaker, ts if ts is not None else pending_ts)
        pending_ts = None

    turns = merge_turns(cues, max_gap_ms=max_gap_ms) if merge else cues
    return Transcript(turns=turns, source=source, format="plain", meta={"cue_count": len(cues)})


def _emit(cues: List[Turn], text: str, speaker: Optional[str], ts: Optional[int]) -> None:
    text = text.strip()
    if not text:
        return
    cues.append(
        Turn(text=text, speaker=speaker, raw_speaker=speaker,
             start_ms=ts, end_ms=None, index=len(cues))
    )
