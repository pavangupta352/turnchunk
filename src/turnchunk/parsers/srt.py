"""SubRip (.srt) parser.

Structurally simpler than WebVTT -- numbered blocks, comma decimal separator,
no voice spans -- but it carries the same inline ``Speaker:`` convention and
the same HTML-ish markup, so it shares the cleaning path.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..types import Transcript, Turn
from .base import (
    drop_rolling_duplicates,
    merge_turns,
    parse_timestamp,
    split_inline_speaker,
    strip_tags,
)

_ARROW = re.compile(r"\s*-->\s*")


def looks_like_srt(text: str) -> bool:
    return bool(
        re.search(r"^\s*\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->", text[:4000], re.MULTILINE)
    )


def parse_srt(
    text: str,
    *,
    source: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
) -> Transcript:
    blocks = re.split(r"\n\s*\n", text.lstrip("﻿").replace("\r\n", "\n").strip())
    cues: List[Turn] = []
    prev_lines: List[str] = []

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].strip().isdigit() and len(lines) > 1:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue

        parts = _ARROW.split(lines[0].strip(), maxsplit=1)
        start = parse_timestamp(parts[0].strip()) if parts else None
        end = parse_timestamp(parts[1].strip().split()[0]) if len(parts) > 1 and parts[1].strip() else None

        cleaned = [strip_tags(ln).strip() for ln in lines[1:]]
        cleaned = [c for c in cleaned if c]
        if not cleaned:
            continue

        deduped = drop_rolling_duplicates(prev_lines, cleaned)
        prev_lines = cleaned
        if not deduped:
            continue

        body = " ".join(deduped).strip()
        speaker, rest = split_inline_speaker(body)
        if speaker is not None:
            body = rest
        if not body:
            continue

        cues.append(
            Turn(
                text=body, speaker=speaker, raw_speaker=speaker,
                start_ms=start, end_ms=end, index=len(cues),
            )
        )

    turns = merge_turns(cues, max_gap_ms=max_gap_ms) if merge else cues
    return Transcript(turns=turns, source=source, format="srt", meta={"cue_count": len(cues)})
