"""Diarization sanity checks.

The two ways this feature can fail are not symmetric. A missed warning is a
missed opportunity; a false warning trains people to ignore all warnings. So
half of these tests are about what must *not* be flagged.
"""

from __future__ import annotations

from turnchunk import Turn, diarization_warnings, parse_file


def T(text, spk, i, s=None, e=None):
    return Turn(text=text, speaker=spk, index=i, start_ms=s, end_ms=e)


def kinds(findings):
    return [f.kind for f in findings]


# ------------------------------------------------------ mid-utterance flip ----


def test_flip_with_no_pause_and_lowercase_continuation_is_high_confidence():
    f = diarization_warnings([
        T("We agreed the renewal goes to six percent, not twelve, and", "Priya", 0, 0, 4000),
        T("the incident report gets attached so it reads as goodwill.", "Marcus", 1, 4050, 8000),
        T("Fine by me.", "Priya", 2, 9000, 10000),
    ])
    assert len(f) == 1
    assert f[0].kind == "mid_utterance_flip"
    assert f[0].confidence == "high"
    assert f[0].speakers == ["Priya", "Marcus"]
    assert (f[0].start_index, f[0].end_index) == (0, 1)
    assert f[0].start_ms == 4050


def test_a_real_pause_is_not_a_flip():
    f = diarization_warnings([
        T("We agreed six percent, not twelve, and", "Priya", 0, 0, 4000),
        T("the report gets attached.", "Marcus", 1, 6000, 8000),
    ])
    assert not f


def test_a_finished_sentence_is_not_a_flip_even_with_no_pause():
    """Fast turn-taking between real speakers is normal conversation."""
    f = diarization_warnings([
        T("We agreed six percent.", "Priya", 0, 0, 4000),
        T("The report gets attached.", "Marcus", 1, 4050, 8000),
    ])
    assert not f


def test_lowercase_asr_transcript_still_finds_flips_but_only_medium():
    """Without capitals anywhere, a lowercase start proves nothing."""
    f = diarization_warnings([
        T("we agreed six percent not twelve and", "priya", 0, 0, 4000),
        T("the report gets attached", "marcus", 1, 4050, 8000),
        T("fine by me", "priya", 2, 9000, 10000),
    ])
    assert kinds(f) == ["mid_utterance_flip"]
    assert f[0].confidence == "medium"


def test_no_timestamps_means_no_flip_findings():
    """Pauses cannot be judged without time, so nothing is claimed."""
    f = diarization_warnings([T("and so", "A", 0), T("then we", "B", 1)])
    assert "mid_utterance_flip" not in kinds(f)


def test_same_speaker_boundary_is_never_a_flip():
    f = diarization_warnings([
        T("and so we", "A", 0, 0, 1000),
        T("moved it to Thursday.", "A", 1, 1010, 2000),
    ])
    assert not f


# ---------------------------------------------------------------- flapping ----


def _alternating(pairs):
    return [T(w, s, i, i * 800, i * 800 + 700) for i, (s, w) in enumerate(pairs)]


def test_both_sides_short_and_alternating_is_flapping():
    f = diarization_warnings(_alternating([
        ("A", "yeah so"), ("B", "the thing"), ("A", "is that"),
        ("B", "we need"), ("A", "to move"), ("B", "it thursday"),
    ]))
    flap = [x for x in f if x.kind == "flapping"]
    assert len(flap) == 1
    assert flap[0].speakers == ["A", "B"]
    assert (flap[0].start_index, flap[0].end_index) == (0, 5)


def test_real_back_channel_is_not_flapping():
    """One side long, one side 'mhm' -- that is a person listening, not a diarizer bug."""
    f = diarization_warnings([
        T("So the migration will take a month because of the data volume", "Alice", 0, 0, 5000),
        T("mhm", "Bob", 1, 5100, 5300),
        T("and we need legal sign-off before anything moves region", "Alice", 2, 5400, 10000),
        T("yeah", "Bob", 3, 10100, 10300),
        T("so realistically Thursday is the earliest.", "Alice", 4, 10400, 13000),
        T("ok", "Bob", 5, 13100, 13300),
    ])
    assert "flapping" not in kinds(f)


def test_three_short_alternations_is_below_the_run_threshold():
    f = diarization_warnings(_alternating([("A", "yes"), ("B", "no"), ("A", "maybe")]))
    assert "flapping" not in kinds(f)


def test_flapping_needs_exactly_two_labels():
    """A, B, C, A is three people talking fast, not one voice flapping."""
    f = diarization_warnings(_alternating([("A", "yes"), ("B", "no"), ("C", "maybe"), ("A", "sure")]))
    assert "flapping" not in kinds(f)


# ------------------------------------------------------------------ ghosts ----


def _long_meeting():
    return [
        T("This is a long substantive turn about the migration plan and its timing. " * 3,
          "SPEAKER_00", i, i * 9000, i * 9000 + 8000)
        for i in range(20)
    ]


def test_sparse_generic_label_is_a_ghost():
    turns = [*_long_meeting(), T("uh yes", "SPEAKER_03", 20, 180000, 180500)]
    ghosts = [x for x in diarization_warnings(turns) if x.kind == "ghost_speaker"]
    assert len(ghosts) == 1
    assert ghosts[0].speakers == ["SPEAKER_03"]
    assert ghosts[0].confidence == "low"


def test_a_named_person_who_spoke_once_is_never_a_ghost():
    turns = [*_long_meeting(), T("uh yes", "Dana Okafor", 20, 180000, 180500)]
    assert "ghost_speaker" not in kinds(diarization_warnings(turns))


def test_a_generic_label_with_real_presence_is_not_a_ghost():
    turns = _long_meeting()
    turns += [T("A proper contribution with a full sentence of content in it.", "SPEAKER_03", 20 + i,
                (20 + i) * 9000, (20 + i) * 9000 + 8000) for i in range(3)]
    assert "ghost_speaker" not in kinds(diarization_warnings(turns))


# --------------------------------------------------------------- contracts ----


def test_clean_real_transcript_has_no_findings():
    assert not diarization_warnings(parse_file("examples/renewal-call.vtt"))


def test_labels_are_never_modified():
    """Findings are advisory. The turns come back untouched."""
    turns = [
        T("We agreed six percent, and", "Priya", 0, 0, 4000),
        T("the report gets attached.", "Marcus", 1, 4050, 8000),
    ]
    before = [(t.speaker, t.text) for t in turns]
    diarization_warnings(turns)
    assert [(t.speaker, t.text) for t in turns] == before


def test_findings_serialise():
    f = diarization_warnings([
        T("We agreed six percent, and", "Priya", 0, 0, 4000),
        T("the report gets attached.", "Marcus", 1, 4050, 8000),
    ])
    d = f[0].to_dict()
    assert set(d) == {"kind", "start_index", "end_index", "speakers", "reason", "confidence", "start_ms"}


def test_findings_are_ordered_by_position():
    turns = _long_meeting()
    turns[5] = T("and then the plan was to", "SPEAKER_00", 5, 45000, 53000)
    turns[6] = T("move the whole thing to Thursday.", "SPEAKER_01", 6, 53050, 62000)
    turns.append(T("uh yes", "SPEAKER_03", 20, 180000, 180500))
    f = diarization_warnings(turns)
    assert [x.start_index for x in f] == sorted(x.start_index for x in f)


def test_accepts_a_transcript_or_a_list():
    t = parse_file("examples/renewal-call.vtt")
    assert diarization_warnings(t) == diarization_warnings(t.turns)
