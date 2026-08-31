"""Format detection and dispatch.

``parse()`` never asks the caller what format a transcript is. Making somebody
declare the format is how a "supports 15 formats" library ends up used for one.
Detection runs on content, not on the file extension, because exports are
routinely saved with the wrong suffix.

**``parse()`` never touches the filesystem.** A ``str`` argument is always
transcript *content*, never a path. This matters: applications routinely do
``parse(request.body)`` on user input, and a version of ``parse()`` that reads a
file whenever the string happens to look like a path turns that into arbitrary
file disclosure -- hand it ``/app/config.yaml`` and the contents come back as
"speaker turns". Use :func:`parse_file`, or pass a :class:`pathlib.Path`, when
you actually mean a file.
"""

from __future__ import annotations

import os
import re
from pathlib import PurePath
from typing import Any, List, Optional, Tuple

from ..types import Transcript
from .asr import looks_like_asr_json, parse_asr_json
from .plain import looks_like_plain, parse_plain
from .srt import looks_like_srt, parse_srt
from .vtt import looks_like_vtt, parse_vtt


class UnknownFormatError(ValueError):
    """Raised when no parser recognises the input."""


# Only used to make the error message helpful. Deliberately does not touch the
# filesystem: checking whether the path exists would leak file existence to an
# attacker probing with untrusted input.
_PATH_HINT = re.compile(
    r"^[^\n]{1,4096}\.(?:vtt|srt|txt|json|jsonl|md|sbv|ass|ssa)$", re.IGNORECASE
)


def _looks_like_a_path(text: str) -> bool:
    return bool(_PATH_HINT.match(text.strip()))


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


def parse_file(
    path: Any,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    **kwargs: Any,
) -> Transcript:
    """Read a transcript file from disk and parse it.

    This is the only entry point that touches the filesystem. Pass it a path you
    control -- never a path taken directly from untrusted input.

    Args:
        path: filesystem path to the transcript.
        encoding, errors: passed to :func:`open`.
        **kwargs: forwarded to :func:`parse`.
    """
    with open(path, "r", encoding=encoding, errors=errors) as fh:
        text = fh.read()
    transcript = parse(text, **kwargs)
    # Record the filename so chunk ids and citations can name their source.
    transcript.source = os.path.basename(os.fspath(path))
    for turn in transcript.turns:
        turn.meta.setdefault("source", transcript.source)
    return transcript


def parse(
    source: Any,
    *,
    format: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
    **kwargs: Any,
) -> Transcript:
    """Parse a transcript from text, a :class:`~pathlib.Path`, or a file object.

    A ``str`` argument is **always** treated as transcript content and never as
    a path -- see the module docstring for why. To read a file, use
    :func:`parse_file` or pass a :class:`pathlib.Path`.

    Args:
        source: the transcript text, an open file object, or a ``PurePath``.
        format: force a parser instead of detecting one.
        merge: merge consecutive cues by the same speaker into whole turns.
            Subtitle cues break every few seconds for display reasons; turns are
            what you actually want to chunk on.
        max_gap_ms: start a new turn when the same speaker resumes after this
            much silence.

    Returns:
        A :class:`~turnchunk.types.Transcript`.

    Raises:
        UnknownFormatError: if no parser recognises the content.
    """
    text, name = _read(source)

    chosen = format or detect_format(text)
    if chosen is None:
        hint = ""
        if isinstance(source, str) and _looks_like_a_path(text):
            hint = (
                " This looks like a file path. parse() treats a string as "
                "transcript content, never as a path -- use parse_file() to "
                "read a file."
            )
        raise UnknownFormatError(
            "Could not detect the transcript format. "
            f"Pass format= explicitly (one of: {', '.join(FORMATS)})." + hint
        )
    for fmt_name, _, parser in _DETECTORS:
        if fmt_name == chosen:
            return parser(text, source=name, merge=merge, max_gap_ms=max_gap_ms, **kwargs)
    raise UnknownFormatError(f"Unknown format {chosen!r}. Known: {', '.join(FORMATS)}")


def _read(source: Any) -> Tuple[str, Optional[str]]:
    if hasattr(source, "read"):
        return source.read(), getattr(source, "name", None)
    if isinstance(source, PurePath):
        # A Path object is an unambiguous request for a file, unlike a str.
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), os.path.basename(os.fspath(source))
    if isinstance(source, (bytes, bytearray)):
        return source.decode("utf-8", errors="replace"), None
    if isinstance(source, str):
        # Never the filesystem. See the module docstring.
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
