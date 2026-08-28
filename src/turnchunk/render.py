"""Rendering chunks for an LLM prompt.

Everybody writes this function by hand and everybody gets a detail subtly wrong
-- timestamps in raw milliseconds, speakers dropped from continuation lines, or
overlap text presented as though it were new. Shipping it removes a whole class
of quiet mistakes.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .types import Chunk


def format_timestamp(ms: Optional[int], *, always_hours: bool = False) -> str:
    """Milliseconds to ``H:MM:SS`` / ``MM:SS``. Unknown time renders ``--:--``."""
    if ms is None:
        return "--:--"
    total = max(0, ms) // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def to_context(
    chunk: Chunk,
    *,
    timestamps: bool = True,
    header: bool = True,
    include_overlap: bool = True,
) -> str:
    """Render a chunk as text to put in a prompt.

    Args:
        timestamps: prefix each turn with a readable time.
        header: include a one-line header naming the source and time span.
        include_overlap: keep turns carried over from the previous chunk. Set
            False when concatenating several chunks into one prompt, so the
            model does not see the same sentence twice.
    """
    lines = []
    if header:
        span = f"{format_timestamp(chunk.start_ms)}–{format_timestamp(chunk.end_ms)}"
        where = chunk.source or "transcript"
        lines.append(f"[{where} {span}]")
    for i, turn in enumerate(chunk.turns):
        if not include_overlap and i in chunk.overlap_indices:
            continue
        prefix = f"({format_timestamp(turn.start_ms)}) " if timestamps else ""
        speaker = turn.speaker or "UNKNOWN"
        lines.append(f"{prefix}{speaker}: {turn.text}")
    return "\n".join(lines)


def to_context_all(chunks: Iterable[Chunk], *, separator: str = "\n\n---\n\n", **kw) -> str:
    """Render several chunks into one prompt block, dropping repeated overlap."""
    kw.setdefault("include_overlap", False)
    return separator.join(to_context(c, **kw) for c in chunks)
