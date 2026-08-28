"""Parser tests.

Every case here is a real-world quirk that breaks a naive implementation.
"""

from __future__ import annotations

import pytest

from turnchunk import UnknownFormatError, detect_format, parse

FIX = "tests/fixtures/"


def test_detects_formats_from_content_not_extension():
    assert detect_format(open(FIX + "teams.vtt").read()) == "vtt"
    assert detect_format(open(FIX + "basic.srt").read()) == "srt"
    assert detect_format(open(FIX + "bracketed.txt").read()) == "plain"


def test_srt_is_not_claimed_by_the_vtt_parser():
    """SubRip uses ',' for fractions and WebVTT uses '.'; detection must not confuse them."""
    assert parse(FIX + "basic.srt").format == "srt"


# ------------------------------------------------------------------ VTT ----

def test_teams_voice_spans():
    t = parse(FIX + "teams.vtt")
    assert t.speakers == ["Alice Chen", "Bob Ferreira"]
    assert len(t) == 2, "consecutive cues by one speaker must merge into one turn"
    assert "twelve percent increase" in t.turns[0].text
    assert t.turns[0].start_ms == 1000 and t.turns[0].end_ms == 9200


def test_youtube_rolling_captions_are_deduplicated():
    """Scrolling captions repeat the previous cue; naive parsers double the text."""
    t = parse(FIX + "youtube_rolling.vtt")
    assert t.meta["rolling_duplicate_cues"] == 3
    text = t.turns[0].text
    assert text.count("the first thing we need to") == 1
    assert text.count("talk about is the migration plan") == 1
    assert text.count("how long it actually takes") == 1
    assert text.endswith("because last time it took a month")


def test_inline_word_timestamps_and_entities_are_stripped():
    vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n"
        "<v Alice><00:00:01.500><c>hello</c> there &amp; welcome</v>\n"
    )
    t = parse(vtt, format="vtt")
    assert t.turns[0].text == "hello there & welcome"
    assert t.turns[0].speaker == "Alice"


def test_zoom_inline_speakers_without_breaking_on_colons():
    t = parse(FIX + "zoom.vtt")
    assert t.speakers == ["Priya Raman", "Tom Ableton"]
    last = t.turns[-1]
    assert last.speaker == "Priya Raman"
    # "So here's the thing:" must not be read as a speaker label.
    assert "So here's the thing:" in last.text


def test_cue_settings_after_the_end_timestamp():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000 align:start position:0%\nhello\n"
    t = parse(vtt, format="vtt")
    assert t.turns[0].start_ms == 1000 and t.turns[0].end_ms == 4000


def test_note_and_style_blocks_are_skipped():
    vtt = (
        "WEBVTT\n\nNOTE this is a comment\nspanning two lines\n\n"
        "STYLE\n::cue { color: red }\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v A>real content</v>\n"
    )
    t = parse(vtt, format="vtt")
    assert len(t) == 1 and t.turns[0].text == "real content"


# ------------------------------------------------------------------ SRT ----

def test_srt_strips_markup_and_reads_inline_speakers():
    t = parse(FIX + "basic.srt")
    assert t.speakers == ["Alice", "Bob"]
    assert t.turns[1].text == "Only if neither party gives notice."
    assert t.turns[0].start_ms == 1000


# ---------------------------------------------------------------- plain ----

def test_otter_style_name_then_timestamp_on_its_own_line():
    t = parse(FIX + "otter.txt")
    assert t.speakers == ["Alice Chen", "Bob Ferreira"]
    assert t.turns[0].start_ms == 1000
    assert "twelve percent increase" in t.turns[0].text


def test_bracketed_timestamps_and_same_speaker_merging():
    t = parse(FIX + "bracketed.txt")
    assert len(t) == 2, "Bob's two consecutive lines should be one turn"
    assert t.turns[1].speaker == "Bob"
    assert "historically" in t.turns[1].text


def test_markdown_bold_speakers():
    t = parse("**Alice:** hello there\n**Bob:** hi back\n", format="plain")
    assert t.speakers == ["Alice", "Bob"]


def test_transcript_with_no_timestamps_reports_none():
    t = parse("Alice: hello\nBob: hi\n", format="plain")
    assert t.has_timestamps is False
    assert t.turns[0].start_ms is None, "unknown time must not become 0"


def test_max_gap_starts_a_new_turn():
    vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>first thought</v>\n\n"
        "00:05:00.000 --> 00:05:01.000\n<v A>much later thought</v>\n"
    )
    assert len(parse(vtt, format="vtt")) == 1
    assert len(parse(vtt, format="vtt", max_gap_ms=30_000)) == 2


# ----------------------------------------------------------------- misc ----

def test_unknown_format_raises_with_a_useful_message():
    with pytest.raises(UnknownFormatError) as e:
        parse("just some prose with no structure at all whatsoever")
    assert "format=" in str(e.value)


def test_parse_accepts_path_string_and_file_object():
    a = parse(FIX + "teams.vtt")
    with open(FIX + "teams.vtt") as fh:
        b = parse(fh)
    assert [t.text for t in a] == [t.text for t in b]


def test_bom_is_tolerated():
    t = parse("﻿WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>hi</v>\n")
    assert len(t) == 1


# ------------------------------------------------- real-world regressions ----
#
# These reproduce structures observed in genuine exports (fetched with yt-dlp
# from public videos). The content is written here rather than copied, but the
# shape -- cue settings, 10ms flush cues, rolling repeats, per-word timings,
# escaped markup -- is exactly what real files contain.


def test_youtube_auto_captions_real_shape():
    """A faithful reproduction of YouTube's auto-caption structure.

    Verified against a real 171 KB export: 498 cues collapsed to one turn with
    978 rolling duplicates removed and zero repeated text.
    """
    t = parse(FIX + "youtube_auto_real_shape.vtt")
    text = t.turns[-1].text
    for phrase in ("the first thing to understand",
                   "is that the index is stale",
                   "and nobody rebuilt it"):
        assert text.count(phrase) == 1, f"{phrase!r} was duplicated by rolling captions"
    assert "<c>" not in text and "00:00:" not in text, "per-word timing markup leaked"
    assert t.meta["rolling_duplicate_cues"] > 0


def test_blank_and_whitespace_only_cues_are_dropped():
    """Auto-captions emit 10ms cues containing a single space."""
    t = parse(FIX + "youtube_auto_real_shape.vtt")
    assert all(x.text.strip() for x in t.turns)


def test_html_escaped_formatting_tags_are_stripped():
    """YouTube writes &lt;i&gt; to mean italics; it must not survive as text.

    Found by running the parser over a real caption file, where the output
    contained a literal '<i>' because entities were decoded after tag removal.
    """
    t = parse(FIX + "escaped_tags.vtt")
    joined = " ".join(x.text for x in t.turns)
    assert "<i>" not in joined and "<b>" not in joined and "&lt;" not in joined
    assert "Softly, in the background" in joined
    assert "Loud and clear" in joined


def test_angle_brackets_that_are_not_tags_survive():
    """Stripping must not eat ordinary text containing angle brackets."""
    t = parse(FIX + "escaped_tags.vtt")
    joined = " ".join(x.text for x in t.turns)
    assert "a < b" in joined
    assert "<div> element stays" in joined
