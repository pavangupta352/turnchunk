"""Speaker identity resolution.

One transcript routinely refers to the same person several ways::

    "John"      "john"      "JOHN SMITH"     "John Smith"
    "J. Smith"  "John S."   "John (Host)"    "SPEAKER_01"

Left alone, that becomes four "different" people in your metadata, so filtering
by participant silently returns a third of their turns and any per-speaker
aggregate is wrong.

The rules here are deliberately conservative, because the two failure modes are
not symmetric: leaving one person as two labels is untidy, but *merging two
different people* silently corrupts attribution -- and attributing a commitment
to the wrong person is the worst thing this library could do. So a merge only
happens when it is unambiguous, and the full mapping is always returned for
inspection.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Turn, turns_from

# "John (Host)", "John [interviewer]", "John - Acme Corp"
_TRAILING_ROLE = re.compile(r"\s*[\(\[\{].*?[\)\]\}]\s*$|\s+[-–—]\s+.*$")
# Generic diarization labels: SPEAKER_00, Speaker 1, spk0, S1
_GENERIC = re.compile(r"^(?:speaker|spk|s)[\s_\-]*(\d+)$", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def canonical_key(name: str) -> str:
    """Fold a label to a comparison key: accents, case and punctuation removed."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT.sub(" ", s)
    return " ".join(s.lower().split())


def is_generic_label(name: str) -> bool:
    """Is this a diarizer's placeholder rather than a human name?"""
    return bool(_GENERIC.match(name.strip()))


def strip_role(name: str) -> str:
    """Remove a trailing role or affiliation: ``"John (Host)"`` -> ``"John"``."""
    cleaned = _TRAILING_ROLE.sub("", name.strip()).strip()
    return cleaned or name.strip()


def _tokens(key: str) -> List[str]:
    return [t for t in key.split() if t]


def _is_initial_form(short: Sequence[str], full: Sequence[str]) -> bool:
    """Does ``short`` look like an abbreviation of ``full``?

    Matches "j smith" against "john smith", and "john s" against "john smith",
    but never "j smith" against "jane smith" *and* "john smith" at once -- that
    ambiguity is resolved by the caller, which refuses to merge.
    """
    if len(short) != len(full) or not short:
        return False
    for a, b in zip(short, full):
        if a == b:
            continue
        if len(a) == 1 and b.startswith(a):
            continue
        return False
    return True


def _compatible(short_key: str, full_key: str) -> bool:
    """Could ``short_key`` be the same person as the longer ``full_key``?"""
    s, f = _tokens(short_key), _tokens(full_key)
    if not s or not f or s == f:
        return False
    if len(s) > len(f):
        return False
    if _is_initial_form(s, f):
        return True
    # "john" ⊂ "john smith" -- a strict subsequence of the full name's tokens.
    return len(s) < len(f) and all(t in f for t in s)


def build_speaker_map(
    names: Iterable[str],
    *,
    counts: Optional[Dict[str, int]] = None,
    merge_partial_names: bool = True,
) -> Dict[str, str]:
    """Map every raw label to its resolved display name.

    Args:
        names: the raw speaker labels as they appeared in the file.
        counts: how often each label occurred, used to pick the display form.
        merge_partial_names: merge "John" into "John Smith" when exactly one
            longer name is compatible. Turn this off to only fold case and
            punctuation variants.

    Returns:
        ``{raw_label: resolved_name}`` covering every input label.
    """
    raw = list(dict.fromkeys(n for n in names if n))
    if not raw:
        return {}
    counts = counts or {}

    # Pass 1 -- fold case, accents, punctuation and trailing roles.
    groups: Dict[str, List[str]] = {}
    for name in raw:
        groups.setdefault(canonical_key(strip_role(name)), []).append(name)

    display: Dict[str, str] = {}
    for key, members in groups.items():
        display[key] = _pick_display(members, counts)

    # Pass 2 -- merge partial names into their unambiguous full form.
    if merge_partial_names:
        keys = list(groups)
        # Generic diarizer labels are never merged into each other: SPEAKER_00
        # and SPEAKER_01 are different people by definition.
        human = [k for k in keys if not is_generic_label(display[k])]
        alias: Dict[str, str] = {}
        for short in human:
            candidates = [f for f in human if f != short and _compatible(short, f)]
            # Only merge when there is exactly one possible match. "J. Smith"
            # with both "John Smith" and "Jane Smith" present stays separate.
            if len(candidates) == 1:
                alias[short] = candidates[0]
        # Resolve chains ("j s" -> "j smith" -> "john smith") without cycling.
        for short in list(alias):
            seen = {short}
            target = alias[short]
            while target in alias and target not in seen:
                seen.add(target)
                target = alias[target]
            alias[short] = target
        for short, target in alias.items():
            display[short] = display[target]

    return {
        name: display[canonical_key(strip_role(name))]
        for name in raw
    }


def _pick_display(members: Sequence[str], counts: Dict[str, int]) -> str:
    """Choose the label a human would recognise.

    Prefers a properly-capitalised form over ALL CAPS or all lowercase, then the
    most frequent, then the longest -- so a file that says "JOHN SMITH" once and
    "John Smith" forty times reports "John Smith".
    """
    def score(name: str) -> Tuple[int, int, int]:
        cleaned = strip_role(name)
        titlecase = int(cleaned != cleaned.upper() and cleaned != cleaned.lower())
        return (titlecase, counts.get(name, 0), len(cleaned))

    return strip_role(max(members, key=score))


def resolve_speakers(
    turns: object,
    *,
    merge_partial_names: bool = True,
    rename: Optional[Dict[str, str]] = None,
) -> Tuple[List[Turn], Dict[str, str]]:
    """Unify speaker labels across a transcript.

    Args:
        turns: a Transcript or list of Turns.
        merge_partial_names: see :func:`build_speaker_map`.
        rename: explicit overrides applied last, e.g.
            ``{"SPEAKER_00": "Alice Chen"}``. Explicit always wins over inferred.

    Returns:
        ``(turns, mapping)`` -- the turns have ``speaker`` resolved and
        ``raw_speaker`` preserved, and the mapping is the audit trail.
    """
    items = turns_from(turns)
    counts: Dict[str, int] = {}
    for t in items:
        name = t.raw_speaker or t.speaker
        if name:
            counts[name] = counts.get(name, 0) + 1

    mapping = build_speaker_map(counts.keys(), counts=counts,
                                merge_partial_names=merge_partial_names)
    if rename:
        # Overrides apply to both the raw label and its resolved form.
        for raw, resolved in list(mapping.items()):
            if raw in rename:
                mapping[raw] = rename[raw]
            elif resolved in rename:
                mapping[raw] = rename[resolved]
        for raw, target in rename.items():
            mapping.setdefault(raw, target)

    for t in items:
        original = t.raw_speaker or t.speaker
        if original is None:
            continue
        t.raw_speaker = original
        t.speaker = mapping.get(original, original)
    return items, mapping
