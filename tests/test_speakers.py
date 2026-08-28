"""Speaker identity resolution.

The asymmetry that drives every rule here: failing to merge one person's
variants is untidy, but merging two different people silently misattributes
what they said. So every ambiguous case must stay separate.
"""

from __future__ import annotations

from turnchunk.speakers import (
    build_speaker_map,
    canonical_key,
    is_generic_label,
    resolve_speakers,
    strip_role,
)
from turnchunk.types import Turn


def people(names):
    return len(set(build_speaker_map(names).values()))


def test_case_and_accent_variants_unify():
    assert people(["John", "john", "JOHN"]) == 1
    assert people(["José", "Jose", "JOSÉ"]) == 1


def test_partial_name_merges_into_full_name():
    m = build_speaker_map(["John", "John Smith"])
    assert m["John"] == "John Smith"


def test_initials_merge_when_unambiguous():
    assert build_speaker_map(["J. Smith", "John Smith"])["J. Smith"] == "John Smith"


def test_ambiguous_initials_never_merge():
    """J. Smith with both John Smith and Jane Smith present must stay separate.

    Guessing here would attribute one person's statements to another.
    """
    assert people(["J. Smith", "John Smith", "Jane Smith"]) == 3


def test_generic_diarizer_labels_stay_distinct():
    assert people(["SPEAKER_00", "SPEAKER_01", "Speaker 2", "spk3"]) == 4
    assert is_generic_label("SPEAKER_00") and is_generic_label("Speaker 1")
    assert not is_generic_label("Alice Chen")


def test_roles_and_affiliations_are_stripped():
    assert strip_role("John (Host)") == "John"
    assert strip_role("John Smith - Acme Corp") == "John Smith"
    assert people(["John (Host)", "John Smith - Acme"]) == 1


def test_distinct_people_stay_distinct():
    assert people(["Alice Chen", "Bob Ferreira", "Priya Raman"]) == 3


def test_display_name_prefers_titlecase_then_frequency():
    m = build_speaker_map(["JOHN SMITH", "John Smith"],
                          counts={"JOHN SMITH": 1, "John Smith": 40})
    assert set(m.values()) == {"John Smith"}


def test_resolve_preserves_the_raw_label_as_an_audit_trail():
    turns = [Turn(text="a", speaker="john", index=0),
             Turn(text="b", speaker="John Smith", index=1)]
    out, mapping = resolve_speakers(turns)
    assert out[0].speaker == "John Smith"
    assert out[0].raw_speaker == "john", "the original label must survive"
    assert mapping["john"] == "John Smith"


def test_explicit_rename_wins_over_inference():
    turns = [Turn(text="a", speaker="SPEAKER_00", index=0)]
    out, _ = resolve_speakers(turns, rename={"SPEAKER_00": "Alice Chen"})
    assert out[0].speaker == "Alice Chen"


def test_merging_can_be_disabled():
    m = build_speaker_map(["John", "John Smith"], merge_partial_names=False)
    assert len(set(m.values())) == 2


def test_canonical_key_folds_punctuation_and_case():
    assert canonical_key("Dr. J. Smith!") == canonical_key("dr j smith")
