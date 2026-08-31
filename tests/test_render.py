"""Prompt rendering."""

from __future__ import annotations

from turnchunk import chunk, parse_file
from turnchunk.render import format_timestamp, to_context, to_context_all

EXAMPLE = "examples/renewal-call.vtt"


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65_000) == "01:05"
    assert format_timestamp(3_725_000) == "1:02:05"


def test_unknown_time_is_visibly_unknown_not_zero():
    assert format_timestamp(None) == "--:--"


def test_to_context_includes_speaker_and_time():
    c = chunk(parse_file(EXAMPLE), target=400)[0]
    text = to_context(c)
    assert "Priya Raman:" in text
    assert "renewal-call.vtt" in text


def test_to_context_can_drop_overlap_to_avoid_repeating_text():
    chunks = chunk(parse_file(EXAMPLE), target=300, overlap=150)
    with_overlap = sum(len(c.overlap_indices) for c in chunks)
    assert with_overlap > 0, "fixture should produce overlap"
    joined = to_context_all(chunks)
    first_turn = chunks[0].turns[0].text[:40]
    assert joined.count(first_turn) == 1, "overlap was repeated into the prompt"
