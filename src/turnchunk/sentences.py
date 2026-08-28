"""Dependency-free sentence segmentation.

A single speaker turn that is longer than the chunk target has to be split
somewhere, and splitting mid-sentence is exactly the damage this library exists
to prevent. That means we need sentence boundaries -- but pulling in nltk or
spacy for it would cost a 500MB install and a model download, which would
destroy the "works anywhere, installs instantly" property.

So this is a hand-written segmenter. It is not a research-grade model and does
not claim to be; it is tuned for the register that actually appears in
transcripts (spoken language, few abbreviations, lots of filler) and it fails
safe: when unsure, it does *not* split, because an over-long chunk is a much
smaller problem than a chunk that begins mid-clause.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Abbreviations whose trailing period is not a sentence end. Kept deliberately
# short: every entry here is a case where we refuse to split, so a wrong entry
# only ever makes chunks longer, never makes them start mid-sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "gen", "col", "lt", "sgt", "capt", "cmdr", "adm", "gov", "pres", "supt",
    "rep", "sen", "atty", "asst", "assoc", "dept", "univ", "inc", "ltd", "co",
    "corp", "llc", "plc", "vs", "etc", "eg", "ie", "al", "approx", "est",
    "fig", "vol", "no", "pp", "ed", "cf", "viz", "ca", "circa",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun",
    "am", "pm", "ph", "phd", "md", "ba", "ma", "bsc", "msc", "llb",
    "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m",
}

# Terminators, optionally followed by closing quotes/brackets.
#
# Latin-script text separates sentences with whitespace; CJK text does not, so
# the whitespace requirement is conditional on which terminator was used.
# Without this split, Japanese and Chinese transcripts are never segmented at
# all and every long turn gets hard-split at an arbitrary character.
_CJK_TERMINATORS = "。！？"
_BOUNDARY = re.compile(
    r"""
    (?P<term>[.!?…。！？]+)          # . ! ? … 。 ！ ？
    (?P<close>["'’”\)\]\}」』]*)   # closing quotes/brackets
    (?P<space>\s*)                                     # required for Latin only
    """,
    re.VERBOSE,
)

# How far back to look when testing for an abbreviation. The patterns are all
# anchored to the end of the window, and the longest entry above is a handful of
# characters, so 40 is generous.
_LOOKBACK = 40

# A token that looks like an initial: "J." in "J. Smith"
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


def _is_abbreviation(before: str) -> bool:
    """Does ``before`` (text up to and including the period) end in an abbreviation?"""
    if _INITIAL.search(before):
        return True
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", before)
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    if word in _ABBREVIATIONS:
        return True
    # Single letter followed by a period inside a run like "U.S." -> treat the
    # whole dotted run as an abbreviation rather than a sentence end.
    return len(word) == 1


def _is_uncapitalised(text: str) -> bool:
    """Does this text look like ASR output with no capitalisation?

    Plenty of speech-to-text produces a single lowercase stream. Requiring a
    capital letter after a period would mean never finding a boundary in those
    transcripts, so we detect the register first and relax the rule when the
    document simply does not use capitals. Detection is per-text rather than
    global because one file can mix a clean vendor transcript with a raw one.
    """
    sample = text[:20_000]
    letters = [c for c in sample if c.isalpha()]
    if len(letters) < 40:
        return False  # too short to judge; stay conservative
    uppers = sum(1 for c in letters if c.isupper())
    return (uppers / len(letters)) < 0.01


def sentence_spans(text: str, *, allow_lowercase_start: Optional[bool] = None) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` spans covering ``text`` exactly.

    Concatenating ``text[s:e]`` for every span reproduces ``text`` byte for
    byte, which lets callers split a turn without ever losing or duplicating a
    character.
    """
    if not text:
        return []

    if allow_lowercase_start is None:
        allow_lowercase_start = _is_uncapitalised(text)

    spans: List[Tuple[int, int]] = []
    start = 0

    # Everything in this loop must be O(1) in the length of the document.
    # Slicing `text[m.end():]` or `text[:m.start()]` here looks harmless and is
    # quadratic: on a 260 KB transcript with 10k boundaries it copied 2.6
    # billion characters and took 1.4 seconds. Only a bounded window is ever
    # needed, so only a bounded window is ever taken.
    for m in _BOUNDARY.finditer(text):
        cut = m.end("close")
        term_start = m.start("term")
        is_cjk = m.group("term")[0] in _CJK_TERMINATORS
        is_period = m.group("term") == "."

        # Latin terminators need whitespace after them, or "example.com" and
        # "3.5x" would both become sentence breaks.
        if not is_cjk and not m.group("space"):
            continue

        next_char = text[m.end()] if m.end() < len(text) else ""

        # "3.14" / "v1.2" -- a period between digits is never a boundary.
        if is_period and next_char.isdigit() and term_start and text[term_start - 1].isdigit():
            continue

        # "Dr. Smith", "e.g. this", "J. Smith". The abbreviation patterns are
        # anchored at the end, so a short trailing window is sufficient.
        if is_period and _is_abbreviation(text[max(start, cut - _LOOKBACK) : cut]):
            continue

        # The next sentence should look like a beginning. Spoken transcripts are
        # lowercase-heavy, so we accept a lowercase start after ! or ?, but
        # require a plausible opener after a bare period.
        if (
            is_period
            and next_char
            and not allow_lowercase_start
            and not _looks_like_start(next_char)
        ):
            continue

        end = m.end("space")
        if end > start:
            spans.append((start, end))
            start = end

    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _looks_like_start(ch: str) -> bool:
    if ch.isupper() or ch.isdigit():
        return True
    if ch in "\"'“‘([{":
        return True
    # Non-cased scripts (CJK, Devanagari, Arabic...) have no capitals, so
    # requiring one would mean never splitting them.
    return bool(ch.isalpha() and ch.lower() == ch.upper())


def split_sentences(text: str, *, allow_lowercase_start: Optional[bool] = None) -> List[str]:
    """Split ``text`` into sentences, preserving all whitespace."""
    return [
        text[s:e]
        for s, e in sentence_spans(text, allow_lowercase_start=allow_lowercase_start)
    ]
