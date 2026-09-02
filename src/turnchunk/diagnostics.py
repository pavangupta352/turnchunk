"""Diarization sanity checks.

A perfectly chunked turn can still carry the wrong name, because the diarizer
that produced the labels made a mistake upstream. turnchunk cannot fix that --
fixing it needs the audio, or at least a model, and being model-free is the
whole reason it installs in a second and runs anywhere.

What it *can* do is notice the symptoms. Diarizer errors leave fingerprints in
the timing and the text that a deterministic pass can find:

* **mid-utterance flip** -- a speaker change with no pause, where the sentence
  visibly continues across the boundary. One person's utterance was cut in two
  and the second half given to someone else.
* **flapping** -- a run of very short turns bouncing between the same two
  labels. Diarizers do this on one voice they cannot settle on. Real
  back-channel ("yeah", "mhm") looks similar but only *one* side is short; here
  both are.
* **ghost speaker** -- a diarizer-generated label (``SPEAKER_03``) with a
  handful of words in an hour of audio. Usually noise. A *named* speaker who
  spoke once is never flagged -- that is a real person.

These are heuristics and are labelled as such. Every finding carries the
turn range, the reason, and a confidence, and nothing is ever changed: the
labels stay exactly as the source gave them, ``raw_speaker`` preserves the
original, and the caller decides. Misattribution is worse than a miss, and a
"fix" that guesses wrong is misattribution with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .speakers import is_generic_label
from .types import Turn, turns_from

_TERMINAL = ".!?…。！？"
_CLOSERS = "\"'’”)]}」』"


@dataclass
class Finding:
    """One suspicious region of a transcript."""

    kind: str
    """``mid_utterance_flip`` | ``flapping`` | ``ghost_speaker``"""
    start_index: int
    """Index of the first turn involved."""
    end_index: int
    """Index of the last turn involved (inclusive)."""
    speakers: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: str = "medium"
    """``high`` | ``medium`` | ``low`` -- how strongly the evidence points."""
    start_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "speakers": list(self.speakers),
            "reason": self.reason,
            "confidence": self.confidence,
            "start_ms": self.start_ms,
        }


def _ends_open(text: str) -> bool:
    """Does the turn end without finishing its sentence?"""
    stripped = text.rstrip().rstrip(_CLOSERS)
    return bool(stripped) and stripped[-1] not in _TERMINAL


def _starts_lower(text: str) -> bool:
    for ch in text.lstrip():
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit():
            return False
    return False


def _word_count(text: str) -> int:
    return len(text.split())


def _uses_capitals(turns: Sequence[Turn]) -> bool:
    """Does this transcript capitalise sentence starts at all?

    Raw ASR output is often one lowercase stream. There, a lowercase start
    means nothing, so the capitalisation signal is only trusted when the
    transcript demonstrably uses capitals elsewhere.
    """
    starts = [t.text.lstrip()[:1] for t in turns if t.text.strip()]
    letters = [c for c in starts if c.isalpha()]
    if len(letters) < 2:
        return False
    # Getting this wrong only moves a finding between "high" and "medium"; it
    # never creates or removes one, so a small sample is acceptable.
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.5


def diarization_warnings(
    turns: Any,
    *,
    max_gap_ms: int = 300,
    flap_max_words: int = 4,
    flap_min_run: int = 4,
    ghost_max_turns: int = 2,
    ghost_max_share: float = 0.005,
) -> List[Finding]:
    """Flag turn boundaries that look like diarizer mistakes.

    Args:
        turns: a Transcript or list of Turns.
        max_gap_ms: a speaker change with less silence than this, where the
            sentence continues, is flagged as a mid-utterance flip.
        flap_max_words: turns at or under this length count toward a flapping
            run.
        flap_min_run: how many consecutive alternating short turns make a
            flapping run.
        ghost_max_turns / ghost_max_share: a *generic* label with no more than
            this many turns and no more than this share of the text is flagged
            as a possible ghost speaker.

    Returns:
        Findings in transcript order. Empty means nothing looked wrong, which
        is not the same as the labels being right.
    """
    items: List[Turn] = list(turns_from(turns))
    findings: List[Finding] = []
    if len(items) < 2:
        return findings

    capitals = _uses_capitals(items)

    # --- mid-utterance flips -------------------------------------------------
    for prev, cur in zip(items, items[1:]):
        if prev.speaker is None or cur.speaker is None or prev.speaker == cur.speaker:
            continue
        if prev.end_ms is None or cur.start_ms is None:
            continue
        gap = cur.start_ms - prev.end_ms
        if gap > max_gap_ms:
            continue
        if not _ends_open(prev.text):
            continue
        # No pause and an unfinished sentence is the core signal; a lowercase
        # continuation on a capitalised transcript makes it near-certain.
        continues = capitals and _starts_lower(cur.text)
        findings.append(
            Finding(
                kind="mid_utterance_flip",
                start_index=prev.index,
                end_index=cur.index,
                speakers=[prev.speaker, cur.speaker],
                confidence="high" if continues else "medium",
                start_ms=cur.start_ms,
                reason=(
                    f"{prev.speaker!r} stops mid-sentence and {cur.speaker!r} "
                    f"starts {max(gap, 0)}ms later"
                    + (" with a lowercase continuation" if continues else "")
                    + " -- likely one utterance split across two labels"
                ),
            )
        )

    # --- flapping ------------------------------------------------------------
    i = 0
    n = len(items)
    while i < n:
        a = items[i].speaker
        if a is None or _word_count(items[i].text) > flap_max_words:
            i += 1
            continue
        # Extend a run of short turns alternating between a and one other label.
        j = i + 1
        b: Optional[str] = None
        while j < n:
            t = items[j]
            expected = b if (j - i) % 2 == 1 and b is not None else None
            if t.speaker is None or _word_count(t.text) > flap_max_words:
                break
            if (j - i) % 2 == 1:
                if b is None:
                    if t.speaker == a:
                        break
                    b = t.speaker
                elif t.speaker != expected:
                    break
            elif t.speaker != a:
                break
            j += 1
        run = j - i
        if b is not None and run >= flap_min_run:
            findings.append(
                Finding(
                    kind="flapping",
                    start_index=items[i].index,
                    end_index=items[j - 1].index,
                    speakers=[a, b],
                    confidence="medium",
                    start_ms=items[i].start_ms,
                    reason=(
                        f"{run} consecutive turns of <= {flap_max_words} words "
                        f"alternating between {a!r} and {b!r} -- diarizers flap "
                        "like this on a single voice; real back-channel has one "
                        "long side"
                    ),
                )
            )
            i = j
        else:
            i += 1

    # --- ghost speakers ------------------------------------------------------
    total_chars = sum(len(t.text) for t in items) or 1
    per_speaker: Dict[str, List[Turn]] = {}
    for t in items:
        if t.speaker is not None:
            per_speaker.setdefault(t.speaker, []).append(t)
    if len(per_speaker) >= 2:
        for name, ts in per_speaker.items():
            if not is_generic_label(name):
                continue  # a named person who spoke once is a real person
            chars = sum(len(t.text) for t in ts)
            if len(ts) <= ghost_max_turns and chars / total_chars <= ghost_max_share:
                words = sum(_word_count(t.text) for t in ts)
                findings.append(
                    Finding(
                        kind="ghost_speaker",
                        start_index=ts[0].index,
                        end_index=ts[-1].index,
                        speakers=[name],
                        confidence="low",
                        start_ms=ts[0].start_ms,
                        reason=(
                            f"{name!r} has {len(ts)} turn(s) and {words} words, "
                            f"{chars / total_chars:.1%} of the transcript -- "
                            "diarizer-generated labels this sparse are often noise"
                        ),
                    )
                )

    findings.sort(key=lambda f: (f.start_index, f.kind))
    return findings


__all__ = ["Finding", "diarization_warnings"]
