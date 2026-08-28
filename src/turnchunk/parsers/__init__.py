"""Format detection and dispatch.

``parse()`` never asks the caller what format a file is. Making somebody
declare the format is how a "supports 15 formats" library ends up used for one.
Detection runs on content, not on the file extension, because exports are
routinely saved with the wrong suffix.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from ..types import Transcript
from .asr import looks_like_asr_json, parse_asr_json
from .plain import looks_like_plain, parse_plain
from .srt import looks_like_srt, parse_srt
from .vtt import looks_like_vtt, parse_vtt


class UnknownFormatError(ValueError):
    """Raised when no parser recognises the input."""


# Ordered most-specific first: VTT and SRT have unambiguous signatures, plain
# text is the fallback because almost anything can look like it.
_DETECTORS: List[Tuple[str, Any, Any]] = [
    # JSON first: its signature is unambiguous, so it can never be misread as
    # plain text, whereas plain text detection is inherently fuzzy.
    ("json", looks_like_asr_json, parse_asr_json),
    ("vtt", looks_like_vtt, parse_vtt),
    ("srt", looks_like_srt, parse_srt),
    ("plain", looks_like_plain, parse_plain),
]

FORMATS = [name for name, _, _ in _DETECTORS]


def detect_format(text: str) -> Optional[str]:
    """Return the detected format name, or None."""
    for name, detector, _ in _DETECTORS:
        if detector(text):
            return name
    return None


def parse(
    source: Any,
    *,
    format: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
    **kwargs: Any,
) -> Transcript:
    """Parse a transcript from a path, a file object, or a string.

    Args:
        source: a filesystem path, an open file, or the transcript text itself.
        format: force a parser instead of detecting one.
        merge: merge consecutive cues by the same speaker into whole turns.
            Subtitle cues break every few seconds for display reasons; turns are
            what you actually want to chunk on.
        max_gap_ms: start a new turn when the same speaker resumes after this
            much silence.

    Returns:
        A :class:`~turnchunk.types.Transcript`.
    """
    text, name = _read(source)

    chosen = format or detect_format(text)
    if chosen is None:
        raise UnknownFormatError(
            "Could not detect the transcript format. "
            f"Pass format= explicitly (one of: {', '.join(FORMATS)})."
        )
    for fmt_name, _, parser in _DETECTORS:
        if fmt_name == chosen:
            return parser(text, source=name, merge=merge, max_gap_ms=max_gap_ms, **kwargs)
    raise UnknownFormatError(f"Unknown format {chosen!r}. Known: {', '.join(FORMATS)}")


def _read(source: Any) -> Tuple[str, Optional[str]]:
    if hasattr(source, "read"):
        return source.read(), getattr(source, "name", None)
    if isinstance(source, (bytes, bytearray)):
        return source.decode("utf-8", errors="replace"), None
    if isinstance(source, str):
        looks_like_path = "\n" not in source and len(source) < 4096
        if looks_like_path and os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(), os.path.basename(source)
        return source, None
    raise TypeError(f"Cannot read a transcript from {type(source).__name__}")


__all__ = [
    "FORMATS",
    "UnknownFormatError",
    "detect_format",
    "parse",
    "parse_asr_json",
    "parse_plain",
    "parse_srt",
    "parse_vtt",
]
