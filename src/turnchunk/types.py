"""Core data types.

A transcript, once parsed, is a list of :class:`Turn`. Chunking a list of turns
produces a list of :class:`Chunk`. Both are plain dataclasses with no behaviour
beyond convenience properties, so they are trivial to serialise, diff and test.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """One uninterrupted stretch of speech by one speaker.

    ``speaker`` is the resolved identity (see :mod:`turnchunk.speakers`);
    ``raw_speaker`` is the label exactly as it appeared in the source file, kept
    so callers can always trace a turn back to what was written on disk.

    Timestamps are milliseconds from the start of the recording, or ``None``.
    ``None`` means "this format did not tell us" and must never be silently
    turned into ``0`` -- downstream users need to distinguish "at the start" from
    "unknown".
    """

    text: str
    speaker: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    index: int = 0
    raw_speaker: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_ms is None or self.end_ms is None:
            return None
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A group of consecutive turns, sized for embedding or prompting.

    ``turns`` holds the turns that make up the chunk, including any carried over
    from the previous chunk as overlap. Overlap turns are listed in
    ``overlap_indices`` so aggregate statistics can exclude them -- counting a
    speaker twice because they appear in two chunks' overlap is a real and
    commonly-shipped bug.
    """

    text: str
    turns: List[Turn] = field(default_factory=list)
    overlap_indices: List[int] = field(default_factory=list)
    source: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- identity ---------------------------------------------------------

    @property
    def id(self) -> str:
        """Stable content-addressed id.

        Derived from the source name, the turn range and the text, so the same
        input always produces the same id across runs and machines -- which is
        what makes chunk ids usable as primary keys in a vector store.
        """
        h = hashlib.sha256()
        h.update((self.source or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(f"{self.turn_start}:{self.turn_end}".encode("utf-8"))
        h.update(b"\x00")
        h.update(self.text.encode("utf-8"))
        return h.hexdigest()[:16]

    # -- speakers ---------------------------------------------------------

    @property
    def speakers(self) -> List[str]:
        """Distinct speakers in the chunk, in order of first appearance."""
        seen: List[str] = []
        for t in self.turns:
            if t.speaker is not None and t.speaker not in seen:
                seen.append(t.speaker)
        return seen

    @property
    def primary_speaker(self) -> Optional[str]:
        """The speaker contributing the most characters, ignoring overlap.

        Overlap is excluded so that a chunk is attributed to whoever actually
        dominates its own content, not to someone carried in from the chunk
        before it.
        """
        totals: Dict[str, int] = {}
        for i, t in enumerate(self.turns):
            if i in self.overlap_indices or t.speaker is None:
                continue
            totals[t.speaker] = totals.get(t.speaker, 0) + len(t.text)
        if not totals:
            return self.turns[0].speaker if self.turns else None
        return max(totals.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # -- position ---------------------------------------------------------

    @property
    def turn_start(self) -> int:
        return self.turns[0].index if self.turns else -1

    @property
    def turn_end(self) -> int:
        return self.turns[-1].index if self.turns else -1

    @property
    def start_ms(self) -> Optional[int]:
        for t in self.turns:
            if t.start_ms is not None:
                return t.start_ms
        return None

    @property
    def end_ms(self) -> Optional[int]:
        for t in reversed(self.turns):
            if t.end_ms is not None:
                return t.end_ms
        return None

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "speakers": self.speakers,
            "primary_speaker": self.primary_speaker,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "turn_start": self.turn_start,
            "turn_end": self.turn_end,
            "overlap_indices": list(self.overlap_indices),
            "source": self.source,
            "meta": dict(self.meta),
        }


@dataclass
class Transcript:
    """Parsed transcript: turns plus whatever the parser learned along the way."""

    turns: List[Turn] = field(default_factory=list)
    source: Optional[str] = None
    format: Optional[str] = None
    speaker_map: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __getitem__(self, i):
        return self.turns[i]

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for t in self.turns:
            if t.speaker is not None and t.speaker not in seen:
                seen.append(t.speaker)
        return seen

    @property
    def has_timestamps(self) -> bool:
        return any(t.start_ms is not None for t in self.turns)

    @property
    def duration_ms(self) -> Optional[int]:
        ends = [t.end_ms for t in self.turns if t.end_ms is not None]
        return max(ends) if ends else None

    @property
    def text(self) -> str:
        return "\n".join(t.text for t in self.turns)


def turns_from(obj: Any) -> List[Turn]:
    """Accept a Transcript, a list of Turns, or a list of dicts."""
    if isinstance(obj, Transcript):
        return obj.turns
    items = list(obj)
    if items and isinstance(items[0], dict):
        return [Turn(**d) for d in items]
    return items  # type: ignore[return-value]
