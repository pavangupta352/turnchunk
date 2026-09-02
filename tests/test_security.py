"""Security and resource-exhaustion regressions.

Every case here was a real defect found by attacking the library with hostile
input after it was already published. None of them were caught by the
correctness suite, because none of them produce a wrong *answer* -- they
produce a wrong *capability*.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from turnchunk import UnknownFormatError, chunk, parse, parse_file
from turnchunk.speakers import MAX_MERGE_LABELS, build_speaker_map, resolve_speakers

# --------------------------------------------------- file disclosure ----
#
# parse() used to read a file whenever the string it was handed happened to be
# a valid path. Applications routinely call parse(request.body) on user input,
# which turned that into arbitrary local file disclosure: hand it a config file
# and the contents came back as "speaker turns".


def test_parse_never_reads_the_filesystem(tmp_path):
    secret = tmp_path / "secret.vtt"
    secret.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>CLASSIFIED</v>\n")

    with pytest.raises(UnknownFormatError):
        parse(str(secret))  # a path *string* is content, and this is not a transcript


def test_parse_does_not_leak_file_contents_even_when_the_file_would_parse(tmp_path):
    secret = tmp_path / "secret.vtt"
    secret.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>CLASSIFIED</v>\n")
    try:
        result = parse(str(secret))
    except UnknownFormatError:
        return
    assert "CLASSIFIED" not in result.text, "file contents leaked through parse()"


def test_the_error_message_points_at_parse_file(tmp_path):
    with pytest.raises(UnknownFormatError, match="parse_file"):
        parse("/some/where/meeting.vtt")


def test_error_path_does_not_probe_the_filesystem(tmp_path):
    """The hint must not reveal whether a path exists."""
    real = tmp_path / "real.vtt"
    real.write_text("not a transcript at all")
    missing = str(tmp_path / "missing.vtt")

    with pytest.raises(UnknownFormatError) as a:
        parse(str(real))
    with pytest.raises(UnknownFormatError) as b:
        parse(missing)
    assert str(a.value) == str(b.value), "error text distinguishes existing from missing files"


def test_parse_file_is_the_explicit_way_in(tmp_path):
    p = tmp_path / "m.vtt"
    p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Alice>hello</v>\n")
    t = parse_file(p)
    assert t.turns[0].text == "hello"
    assert t.source == "m.vtt"


def test_a_path_object_is_unambiguous_and_still_works(tmp_path):
    p = tmp_path / "m.vtt"
    p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Alice>hello</v>\n")
    assert parse(Path(p)).turns[0].text == "hello"


# ------------------------------------------- speaker-resolution DoS ----
#
# build_speaker_map compared every label against every other. The plain-text
# parser treats any "Word: text" line as a speaker, so an uploaded log or YAML
# file could manufacture thousands of them -- 6,000 lines cost 13.7 seconds of
# CPU from a single request.


def test_many_distinct_speakers_does_not_burn_cpu():
    hostile = "\n".join(f"Key{i}Unique{i}: value number {i}" for i in range(6000))
    t = parse(hostile, format="plain")
    assert len(t.speakers) > 1000, "fixture should manufacture many fake speakers"

    start = time.perf_counter()
    resolve_speakers(t.turns)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"speaker resolution took {elapsed:.1f}s on hostile input"


def test_speaker_resolution_scales_linearly():
    """Catches a return to quadratic behaviour without flaking on shared runners.

    A single ~3ms measurement is dominated by scheduler jitter on a busy CI
    box, and one interruption in the small sample blew the ratio to 14x on a
    macOS runner while the same commit passed everywhere else. So: baselines
    large enough that jitter is a small fraction, and best-of-N, because an
    interruption can only ever make a run slower -- the minimum is the honest
    number. Quadratic code still shows ~16x for 4x input; linear sits near 1.
    """
    def timed(n, repeats=5):
        names = [f"Person{i} Surname{i}" for i in range(n)]
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            build_speaker_map(names, max_merge_labels=10**9)  # force the merge path
            best = min(best, time.perf_counter() - start)
        return best

    small, large = timed(1000), timed(4000)  # 4x the labels
    if small <= 0:
        return
    assert (large / small) / 4 < 3.0, (
        f"speaker resolution scales super-linearly: {small*1000:.1f}ms -> {large*1000:.1f}ms"
    )


def test_merging_is_skipped_above_the_cap_but_folding_still_works():
    names = [f"Person{i}" for i in range(MAX_MERGE_LABELS + 50)] + ["alice", "Alice"]
    mapping = build_speaker_map(names)
    assert mapping["alice"] == mapping["Alice"], "case folding must survive the cap"


def test_the_cap_does_not_change_small_transcripts():
    """Real conversations are far below the cap and must behave identically."""
    names = ["John", "John Smith", "J. Smith", "Dana Okafor"]
    assert build_speaker_map(names) == build_speaker_map(names, max_merge_labels=10**9)


# ------------------------------------------------ malformed numbers ----


@pytest.mark.parametrize(
    "payload",
    [
        '[{"text":"hi","speaker":"A","start":NaN,"end":Infinity}]',
        '[{"text":"hi","speaker":"A","start":-Infinity,"end":1}]',
        '{"recognizedPhrases":[{"speaker":1,"offsetInTicks":Infinity,'
        '"durationInTicks":NaN,"nBest":[{"display":"hi"}]}]}',
        '{"results":[{"alternatives":[{"transcript":"x","words":['
        '{"startTime":{"seconds":Infinity},"word":"hi","speakerTag":1}]}]}]}',
    ],
)
def test_nan_and_infinity_do_not_crash(payload):
    """Python's json module accepts NaN/Infinity; round() raises OverflowError.

    An unexpected exception type from a parser is worse than a wrong number --
    callers cannot defend against it.
    """
    t = parse(payload)
    for turn in t.turns:
        assert turn.start_ms is None or isinstance(turn.start_ms, int)
        assert turn.end_ms is None or isinstance(turn.end_ms, int)


def test_hostile_inputs_never_raise_an_unexpected_exception_type():
    hostile = [
        "", "   \n\t ", "WEBVTT\n", "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n",
        "WEBVTT\n\n00:00:09.000 --> 00:00:01.000\n<v A>backwards</v>",
        "WEBVTT\n\n999:59:59.999 --> 999:59:59.999\n<v A>far</v>",
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>ha\x00llo</v>",
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v >empty</v>",
        '{"segments":null}', '{"segments":"nope"}', '{"segments":[{"text":"hi"',
        json.dumps([{"text": "hi", "speaker": {"a": 1}, "start": 0}]),
        json.dumps([{"text": "hi", "speaker": [1, 2], "start": "soon"}]),
    ]
    for payload in hostile:
        try:
            t = parse(payload)
            chunk(t, target=200)
        except (UnknownFormatError, ValueError, TypeError):
            pass  # typed, catchable, documented
        except Exception as exc:
            pytest.fail(f"{payload[:40]!r} raised {type(exc).__name__}: {exc}")
