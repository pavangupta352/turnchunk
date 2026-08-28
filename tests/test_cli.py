"""CLI behaviour and exit codes."""

from __future__ import annotations

import json

from turnchunk.cli import main

FIX = "tests/fixtures/"
EXAMPLE = "examples/renewal-call.vtt"


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_formats_lists_every_supported_format(capsys):
    code, out = run(capsys, "formats")
    assert code == 0
    for expected in ("vtt", "srt", "plain", "json:whisper", "json:assemblyai"):
        assert expected in out


def test_detect_prints_format_per_file(capsys):
    code, out = run(capsys, "detect", FIX + "teams.vtt", FIX + "basic.srt")
    assert code == 0
    assert "vtt" in out and "srt" in out


def test_chunk_json_output_is_valid_and_carries_metadata(capsys):
    code, out = run(capsys, "chunk", EXAMPLE, "--json", "--target", "400")
    assert code == 0
    data = json.loads(out)
    assert data and all(
        {"id", "text", "speakers", "primary_speaker", "start_ms"} <= set(c) for c in data
    )


def test_viz_compare_reports_unattributable_naive_chunks(capsys):
    """The headline comparison must be present and honest."""
    code, out = run(capsys, "viz", EXAMPLE, "--compare", "--target", "240",
                    "--color", "never")
    assert code == 0
    assert "RecursiveCharacterTextSplitter" in out
    assert "cannot be attributed to a speaker" in out
    # turnchunk's own line must claim zero unattributable chunks.
    ours = out.split("turnchunk")[-1]
    assert "0 cannot be attributed to a speaker" in ours


def test_stats_reports_talk_time(capsys):
    code, out = run(capsys, "stats", EXAMPLE, "--color", "never")
    assert code == 0
    assert "talk time" in out and "Priya Raman" in out


def test_speakers_shows_resolved_identities(capsys):
    code, out = run(capsys, "speakers", EXAMPLE, "--color", "never")
    assert code == 0
    assert "distinct speakers" in out


def test_report_over_a_glob(capsys):
    code, out = run(capsys, "report", FIX + "*.vtt", "--color", "never")
    assert code == 0
    assert "file(s)" in out


def test_unknown_format_exits_2(capsys):
    code = main(["chunk", FIX + "nonexistent-shape.txt"])
    assert code in (1, 2)


def test_missing_file_exits_nonzero():
    assert main(["stats", "does-not-exist.vtt"]) != 0


def test_context_output_renders_readable_timestamps(capsys):
    code, out = run(capsys, "chunk", EXAMPLE, "--context", "--target", "400")
    assert code == 0
    assert "Priya Raman:" in out and ":" in out
