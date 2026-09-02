"""turnchunk -- chunking that understands who is speaking.

    from turnchunk import parse, chunk

    turns  = parse("standup.vtt")            # format auto-detected
    chunks = chunk(turns, target=2000, overlap=200)

    for c in chunks:
        print(c.primary_speaker, c.start_ms, c.text[:80])

Zero dependencies. Every chunk boundary falls on a speaker turn boundary.
"""

from __future__ import annotations

from .chunker import chunk, render_turn, split_oversized_turn
from .diagnostics import Finding, diarization_warnings
from .parsers import (
    FORMATS,
    UnknownFormatError,
    detect_format,
    parse,
    parse_file,
)
from .sentences import split_sentences
from .types import Chunk, Transcript, Turn

__version__ = "0.4.0"

__all__ = [
    "FORMATS",
    "Chunk",
    "Finding",
    "Transcript",
    "Turn",
    "UnknownFormatError",
    "__version__",
    "chunk",
    "detect_format",
    "diarization_warnings",
    "parse",
    "parse_file",
    "render_turn",
    "split_oversized_turn",
    "split_sentences",
]
