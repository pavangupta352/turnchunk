"""The Python side of the cross-language conformance contract.

`tests/corpus/conformance.json` is generated from this implementation, so these
assertions are a regression guard: if a refactor changes any output, the corpus
stops matching and the TypeScript port -- which is verified against the same
file -- would silently drift.

Regenerate deliberately with `python scripts/build_corpus.py`, and review the
diff. An unexplained diff means a behaviour change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnchunk import Turn, chunk, diarization_warnings, parse_file
from turnchunk.sentences import split_sentences
from turnchunk.speakers import build_speaker_map

ROOT = Path(__file__).resolve().parent.parent
CORPUS = json.loads((ROOT / "tests" / "corpus" / "conformance.json").read_text("utf-8"))
FIXTURES = ROOT / "tests" / "fixtures"


def test_corpus_is_present_and_non_trivial():
    assert CORPUS["version"] == 1
    assert len(CORPUS["parse"]) > 10
    assert sum(len(c["chunks"]) for v in CORPUS["chunk"].values() for c in v) > 30


@pytest.mark.parametrize("name", sorted(CORPUS["parse"]))
def test_parse_matches_corpus(name):
    expected = CORPUS["parse"][name]
    t = parse_file(FIXTURES / name)
    assert t.format == expected["format"]
    got = [
        {
            "text": x.text, "speaker": x.speaker, "start_ms": x.start_ms,
            "end_ms": x.end_ms, "index": x.index,
        }
        for x in t.turns
    ]
    assert got == expected["turns"]


@pytest.mark.parametrize("name", sorted(CORPUS["chunk"]))
def test_chunk_matches_corpus(name):
    t = parse_file(FIXTURES / name)
    for case in CORPUS["chunk"][name]:
        chunks = chunk(t, source=name, **case["options"])
        got = [
            {
                "id": c.id, "text": c.text, "speakers": c.speakers,
                "primary_speaker": c.primary_speaker, "start_ms": c.start_ms,
                "end_ms": c.end_ms, "turn_start": c.turn_start,
                "turn_end": c.turn_end, "overlap_indices": list(c.overlap_indices),
            }
            for c in chunks
        ]
        assert got == case["chunks"], f"{name} with {case['options']}"


def test_sentences_match_corpus():
    for case in CORPUS["sentences"]:
        assert split_sentences(case["input"]) == case["output"]


def test_speakers_match_corpus():
    for case in CORPUS["speakers"]:
        assert build_speaker_map(case["input"]) == case["output"]


def test_diagnostics_match_corpus():
    for case in CORPUS["diagnostics"]:
        got = [
            {
                "kind": f.kind, "start_index": f.start_index, "end_index": f.end_index,
                "speakers": f.speakers, "confidence": f.confidence,
                "start_ms": f.start_ms, "end_ms": f.end_ms,
            }
            for f in diarization_warnings([Turn(**d) for d in case["input"]])
        ]
        assert got == case["output"]
