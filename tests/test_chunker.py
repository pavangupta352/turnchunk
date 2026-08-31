"""Tests for the four chunking rules.

The headline test is ``test_a_turn_is_never_split`` -- it is the library's
entire claim, so it runs as a property test across hundreds of generated
transcripts and every reasonable configuration rather than against one fixture.
"""

from __future__ import annotations

import random

import pytest

from turnchunk import Turn, chunk, render_turn
from turnchunk.chunker import DEFAULT_SEPARATOR, DEFAULT_TEMPLATE

SPEAKERS = ["Alice", "Bob Ferreira", "Priya Raman", "SPEAKER_00", "Dr. Chen"]

SENTENCES = [
    "The contract renews automatically unless notice is given. ",
    "I'd want to see their justification before we agree. ",
    "Twelve percent is well above what we budgeted for this year. ",
    "Can we circle back on that next week? ",
    "Right. ",
    "Mhm. ",
    "So the migration took about a month last time, maybe a little more. ",
]


def make_turns(rng: random.Random, n: int, *, long_turn_chance: float = 0.12):
    turns = []
    for i in range(n):
        count = rng.randint(1, 4)
        if rng.random() < long_turn_chance:
            count = rng.randint(30, 60)  # a monologue that must be split
        text = "".join(rng.choice(SENTENCES) for _ in range(count)).strip()
        turns.append(
            Turn(
                text=text,
                speaker=rng.choice(SPEAKERS),
                start_ms=i * 5000,
                end_ms=i * 5000 + 4000,
                index=i,
            )
        )
    return turns


def rendered_lines(c):
    return c.text.split(DEFAULT_SEPARATOR)


# ---------------------------------------------------------------- rule 2 ----

def test_a_turn_is_never_split():
    """THE invariant: a turn that fits the target is never cut across chunks.

    Any turn small enough to fit must appear in some chunk as one complete,
    contiguous rendered line. If a chunk boundary ever landed inside a turn, the
    turn's rendered text would appear as two partial lines and this assertion
    would fail.
    """
    rng = random.Random(20260828)
    configs = [
        {"target": t, "overlap": o, "min_tail_ratio": r}
        for t in (200, 500, 2000)
        for o in (0, 50, 150)
        for r in (0.0, 1 / 3, 0.5)
        if o < t
    ]
    checked = 0
    for trial in range(40):
        turns = make_turns(rng, rng.randint(3, 40))
        for cfg in configs:
            chunks = chunk(turns, **cfg)
            emitted = set()
            for c in chunks:
                emitted.update(rendered_lines(c))
            for t in turns:
                line = render_turn(t, DEFAULT_TEMPLATE)
                if len(line) <= cfg["target"]:
                    assert line in emitted, (
                        f"turn {t.index} was split across chunks "
                        f"(config={cfg}, trial={trial})"
                    )
                    checked += 1
    assert checked > 5000, f"property test too weak: only {checked} assertions"


def test_chunk_boundaries_only_at_turn_boundaries():
    """Every line of every chunk is a whole rendered turn, never a fragment."""
    rng = random.Random(7)
    turns = make_turns(rng, 60)
    chunks = chunk(turns, target=600, overlap=100)
    valid = set()
    for t in turns:
        for piece in chunk([t], target=600):
            valid.update(rendered_lines(piece))
    for c in chunks:
        for line in rendered_lines(c):
            assert line in valid, f"chunk contains a fragment, not a whole turn: {line[:70]!r}"


def test_no_content_is_lost():
    rng = random.Random(11)
    turns = make_turns(rng, 30, long_turn_chance=0.0)
    chunks = chunk(turns, target=800, overlap=0)
    joined = " ".join(c.text for c in chunks)
    for t in turns:
        assert t.text in joined, f"turn {t.index} vanished"


# ---------------------------------------------------------------- rule 1 ----

def test_oversized_turn_splits_at_sentences_and_keeps_speaker():
    long_text = ("The migration plan needs a full month of runway. " * 300).strip()
    turns = [
        Turn(text="Quick question.", speaker="Alice", index=0, start_ms=0, end_ms=1000),
        Turn(text=long_text, speaker="Bob", index=1, start_ms=1000, end_ms=600_000),
    ]
    chunks = chunk(turns, target=1000)
    pieces = [t for c in chunks for t in c.turns if t.index == 1]
    assert len(pieces) > 5, "monologue was not split"
    assert all(p.speaker == "Bob" for p in pieces), "speaker lost on a split piece"
    for p in pieces:
        assert p.text.strip(), "empty piece emitted"
        # Each piece ends at a sentence end (or is the final fragment).
        assert p.text.rstrip().endswith(".") or p is pieces[-1]
    assert "".join(p.text for p in pieces).replace(" ", "") == long_text.replace(" ", "")


def test_single_sentence_longer_than_target_is_hard_split():
    turns = [Turn(text="word " * 2000, speaker="Alice", index=0)]
    chunks = chunk(turns, target=500)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 500 * 1.5


def test_split_pieces_flag_estimated_timestamps():
    turns = [Turn(text=("A sentence here. " * 200).strip(), speaker="A",
                  index=0, start_ms=0, end_ms=100_000)]
    pieces = [t for c in chunk(turns, target=400) for t in c.turns]
    assert len(pieces) > 1
    assert any(p.meta.get("time_estimated") for p in pieces), \
        "interpolated timestamps must be marked as estimates"


# ---------------------------------------------------------------- rule 4 ----

def test_short_tail_merges_backwards():
    # Sized so the stub cannot fit alongside Bob: it is forced into its own
    # chunk by greedy packing, and only the tail-merge rule can rescue it.
    turns = [
        Turn(text="A" * 480, speaker="Alice", index=0),
        Turn(text="B" * 490, speaker="Bob", index=1),
        Turn(text="ok", speaker="Alice", index=2),  # a 2-char stub
    ]
    merged = chunk(turns, target=500, min_tail_ratio=1 / 3)
    assert not any(len(c.text) < 100 for c in merged), "a stub chunk was emitted"
    assert "ok" in merged[-1].text, "the tail was dropped instead of merged"

    unmerged = chunk(turns, target=500, min_tail_ratio=0.0)
    assert len(unmerged) > len(merged), "min_tail_ratio=0 should keep the stub"


def test_tail_merge_respects_max_overflow():
    turns = [
        Turn(text="A" * 490, speaker="Alice", index=0),
        Turn(text="B" * 490, speaker="Bob", index=1),
        Turn(text="C" * 300, speaker="Alice", index=2),
    ]
    chunks = chunk(turns, target=500, min_tail_ratio=0.9, max_overflow=1.0)
    assert len(chunks) == 3, "merge should be refused when it would overflow"


# ---------------------------------------------------------------- rule 3 ----

def test_overlap_is_whole_turns_and_is_marked():
    rng = random.Random(3)
    turns = make_turns(rng, 40, long_turn_chance=0.0)
    chunks = chunk(turns, target=700, overlap=200)
    assert len(chunks) > 2
    for prev, cur in zip(chunks, chunks[1:]):
        n = len(cur.overlap_indices)
        if n:
            carried = [t.index for t in cur.turns[:n]]
            tail = [t.index for t in prev.turns[-n:]]
            assert carried == tail, "overlap is not a whole-turn suffix of the previous chunk"


def test_overlap_does_not_double_count_the_primary_speaker():
    turns = [
        Turn(text="X" * 400, speaker="Alice", index=0),
        Turn(text="Y" * 400, speaker="Alice", index=1),
        Turn(text="Z" * 60, speaker="Bob", index=2),
    ]
    chunks = chunk(turns, target=500, overlap=400, min_tail_ratio=0.0)
    last = chunks[-1]
    if last.overlap_indices:
        # Bob's own content dominates, even though Alice was carried in.
        assert last.primary_speaker == "Bob"


def test_overlap_never_prevents_progress():
    turns = [Turn(text="word " * 40, speaker=f"S{i}", index=i) for i in range(30)]
    chunks = chunk(turns, target=300, overlap=299)
    assert len(chunks) < 200, "overlap caused near-infinite chunking"
    seen = {t.index for c in chunks for t in c.turns}
    assert seen == set(range(30))


# ------------------------------------------------------------- contracts ----

def test_missing_timestamps_stay_none():
    turns = [Turn(text="hello", speaker="A", index=0),
             Turn(text="hi", speaker="B", index=1)]
    c = chunk(turns, target=100)[0]
    assert c.start_ms is None and c.end_ms is None, "unknown time became 0"


def test_chunk_ids_are_stable_and_distinct():
    rng = random.Random(5)
    turns = make_turns(rng, 20, long_turn_chance=0.0)
    a = chunk(turns, target=600, source="m.vtt")
    b = chunk(turns, target=600, source="m.vtt")
    assert [c.id for c in a] == [c.id for c in b], "ids are not deterministic"
    assert len({c.id for c in a}) == len(a), "id collision"
    c = chunk(turns, target=600, source="other.vtt")
    assert [x.id for x in a] != [x.id for x in c], "source not part of the id"


def test_empty_input():
    assert chunk([]) == []


def test_invalid_config_is_rejected():
    turns = [Turn(text="x", speaker="A", index=0)]
    for kwargs in ({"target": 0}, {"overlap": -1}, {"target": 100, "overlap": 100},
                   {"min_tail_ratio": 1.0}):
        with pytest.raises(ValueError):
            chunk(turns, **kwargs)


def test_token_budget_via_size_fn():
    def words(s):
        return len(s.split())

    turns = [Turn(text=" ".join(["word"] * 50), speaker="A", index=i) for i in range(10)]
    chunks = chunk(turns, target=120, size_fn=words)
    for c in chunks:
        assert words(c.text) <= 120 * 1.5


def test_accepts_dicts_and_transcripts():
    from turnchunk import parse_file
    t = parse_file("tests/fixtures/teams.vtt")
    assert chunk(t, target=200)
    assert chunk([{"text": "hi", "speaker": "A", "index": 0}], target=100)
