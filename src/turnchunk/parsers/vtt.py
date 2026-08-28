"""WebVTT parser.

WebVTT is the format most transcripts arrive in, and almost every producer
writes a different dialect of it:

* **Teams** wraps the speaker in a ``<v Alice>`` voice span.
* **Zoom** writes ``Alice: text`` inside the cue payload and numbers its cues.
* **YouTube auto-captions** scroll: consecutive cues repeat the previous lines
  and add one new one, and they carry per-word ``<00:00:01.234>`` timestamps.
* **Otter/Descript** exports mix numbered cue identifiers with inline names.

All of them are valid WebVTT. A parser that handles only one is the reason
people give up and hand-roll their own.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..types import Transcript, Turn
from .base import (
    drop_rolling_duplicates,
    extract_voice_speaker,
    merge_turns,
    parse_timestamp,
    split_inline_speaker,
    strip_tags,
)

_ARROW = re.compile(r"\s*-->\s*")
_BLOCK_KEYWORDS = ("NOTE", "STYLE", "REGION", "COMMENT")


def looks_like_vtt(text: str) -> bool:
    head = text.lstrip("﻿ \t\r\n")[:64].upper()
    if head.startswith("WEBVTT"):
        return True
    # Some tools omit the WEBVTT header. WebVTT uses "." for fractional seconds
    # and SubRip uses ","; requiring the period here is what keeps a headerless
    # .srt from being claimed by this parser.
    return bool(re.search(r"\d{2}:\d{2}\.\d{3}\s*-->", text[:4000]))


def parse_vtt(
    text: str,
    *,
    source: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
) -> Transcript:
    text = text.lstrip("﻿")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cues: List[Turn] = []
    prev_lines: List[str] = []
    i = 0
    n = len(lines)
    rolling_hits = 0

    while i < n:
        line = lines[i].strip()

        if not line:
            i += 1
            continue
        if line.upper().startswith("WEBVTT"):
            i += 1
            continue
        if any(line.startswith(k) for k in _BLOCK_KEYWORDS):
            # Skip the whole block up to the next blank line.
            while i < n and lines[i].strip():
                i += 1
            continue

        # A cue is: [identifier] / timestamp line / payload lines.
        ts_line = line
        if "-->" not in ts_line:
            if i + 1 < n and "-->" in lines[i + 1]:
                i += 1
                ts_line = lines[i].strip()
            else:
                i += 1
                continue

        start_ms, end_ms = _parse_cue_timing(ts_line)
        i += 1

        payload: List[str] = []
        while i < n and lines[i].strip():
            payload.append(lines[i])
            i += 1

        if not payload:
            continue

        raw = "\n".join(payload)
        speaker = extract_voice_speaker(raw)
        cleaned = [strip_tags(p).strip() for p in payload]
        cleaned = [c for c in cleaned if c]
        if not cleaned:
            continue

        # YouTube-style scrolling captions repeat the tail of the previous cue.
        deduped = drop_rolling_duplicates(prev_lines, cleaned)
        if len(deduped) != len(cleaned):
            rolling_hits += 1
        prev_lines = cleaned
        if not deduped:
            continue

        body = " ".join(deduped).strip()
        if speaker is None:
            inline, rest = split_inline_speaker(body)
            if inline is not None:
                speaker, body = inline, rest
        if not body:
            continue

        cues.append(
            Turn(
                text=body,
                speaker=speaker,
                raw_speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                index=len(cues),
            )
        )

    turns = merge_turns(cues, max_gap_ms=max_gap_ms) if merge else cues
    return Transcript(
        turns=turns,
        source=source,
        format="vtt",
        meta={"cue_count": len(cues), "rolling_duplicate_cues": rolling_hits},
    )


def _parse_cue_timing(ts_line: str):
    parts = _ARROW.split(ts_line, maxsplit=1)
    if len(parts) != 2:
        return None, None
    start = parse_timestamp(parts[0].strip().split()[-1]) if parts[0].strip() else None
    # The end timestamp may be followed by cue settings: "align:start position:0%".
    end_token = parts[1].strip().split()[0] if parts[1].strip() else ""
    end = parse_timestamp(end_token)
    return start, end
