"""chonkie adapter -- the conversation chunker chonkie does not ship.

chonkie has eleven chunkers and none of them understands a speaker turn. This
exposes turnchunk through chonkie's calling convention so it slots into an
existing chonkie pipeline:

    from turnchunk.integrations.chonkie import ConversationChunker

    chunks = ConversationChunker(chunk_size=2000)("meeting.vtt")

chonkie does not need to be installed -- this returns turnchunk's own Chunk
objects, which carry strictly more information than a chonkie chunk does.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..chunker import chunk as _chunk
from ..parsers import parse as _parse
from ..parsers import parse_file as _parse_file
from ..speakers import resolve_speakers
from ..types import Chunk


class ConversationChunker:
    """A chonkie-shaped callable that chunks on speaker turns."""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        *,
        min_tail_ratio: float = 1 / 3,
        resolve_speaker_names: bool = True,
        **chunk_kwargs: Any,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tail_ratio = min_tail_ratio
        self.resolve_speaker_names = resolve_speaker_names
        self.chunk_kwargs = chunk_kwargs

    def chunk(self, text: Any, *, format: Optional[str] = None) -> List[Chunk]:
        """Chunk a transcript.

    Accepts the same inputs as :func:`turnchunk.parse`: a ``str`` is transcript
    *content*, a :class:`pathlib.Path` is a file. Use the ``*_file`` variant to
    read a path given as a string -- never pass untrusted input as a path.
        """
        transcript = _parse(text, format=format)
        if self.resolve_speaker_names:
            resolve_speakers(transcript.turns)
        return _chunk(
            transcript,
            target=self.chunk_size,
            overlap=self.chunk_overlap,
            min_tail_ratio=self.min_tail_ratio,
            **self.chunk_kwargs,
        )

    def chunk_file(self, path: Any, *, format: Optional[str] = None) -> List[Chunk]:
        """Read a transcript file from disk, then chunk it."""
        transcript = _parse_file(path, format=format)
        if self.resolve_speaker_names:
            resolve_speakers(transcript.turns)
        return _chunk(
            transcript,
            target=self.chunk_size,
            overlap=self.chunk_overlap,
            min_tail_ratio=self.min_tail_ratio,
            **self.chunk_kwargs,
        )

    def chunk_batch(self, texts: Sequence[Any]) -> List[List[Chunk]]:
        return [self.chunk(t) for t in texts]

    def __call__(self, text: Any) -> List[Chunk]:
        if isinstance(text, (list, tuple)):
            return self.chunk_batch(text)  # type: ignore[return-value]
        return self.chunk(text)

    def __repr__(self) -> str:
        return "ConversationChunker(chunk_size={}, chunk_overlap={})".format(
            self.chunk_size, self.chunk_overlap
        )
