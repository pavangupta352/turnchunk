"""ASR vendor JSON.

The trap that ruins pipelines silently is time units: most vendors emit seconds,
AssemblyAI emits milliseconds. Reading one as the other leaves the text correct
and every citation pointing at the wrong moment.
"""

from __future__ import annotations

import json

import pytest

from turnchunk import parse_file
from turnchunk.parsers.asr import parse_asr_json

FIX = "tests/fixtures/"


def test_whisperx_diarized_segments():
    t = parse_file(FIX + "whisperx_diarized.json")
    assert t.format == "json:whisper"
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(t) == 2, "consecutive same-speaker segments must merge"
    assert t.turns[0].start_ms == 500 and t.turns[0].end_ms == 9100


def test_plain_whisper_without_diarization():
    t = parse_file(FIX + "whisper_plain.json")
    assert t.turns[0].speaker is None, "absent speaker must stay None, not be invented"
    assert t.turns[0].start_ms == 0 and t.turns[0].end_ms == 6000


def test_assemblyai_times_are_milliseconds_not_seconds():
    """The 1000x bug. 2100 in AssemblyAI is 2.1s, not 35 minutes."""
    t = parse_file(FIX + "assemblyai.json")
    assert t.turns[0].start_ms == 2100
    assert t.turns[0].end_ms == 5400
    assert t.speakers == ["A", "B"]


def test_deepgram_prefers_utterances():
    t = parse_file(FIX + "deepgram.json")
    assert t.format == "json:deepgram"
    assert t.turns[0].start_ms == 500, "seconds must be converted to ms"
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_deepgram_falls_back_to_words_grouped_by_speaker():
    t = parse_file(FIX + "deepgram_words_only.json")
    assert len(t) == 2
    assert t.turns[0].text == "Hello there."
    assert t.turns[0].speaker == "SPEAKER_00"


def test_rev_monologues_join_punctuation_correctly():
    t = parse_file(FIX + "rev.json")
    assert t.turns[0].text == "The contract."
    assert t.turns[0].start_ms == 1020


def test_integer_speaker_ids_become_stable_labels():
    t = parse_file(FIX + "deepgram.json")
    assert t.turns[0].speaker == "SPEAKER_00"


def test_speechmatics_shape():
    payload = json.dumps({"results": [
        {"start_time": 0.5, "end_time": 0.9, "type": "word",
         "alternatives": [{"content": "Hello", "speaker": "S1"}]},
        {"start_time": 0.9, "end_time": 0.9, "type": "punctuation",
         "alternatives": [{"content": ".", "speaker": "S1"}]},
        {"start_time": 1.5, "end_time": 2.0, "type": "word",
         "alternatives": [{"content": "Hi", "speaker": "S2"}]}]})
    t = parse_asr_json(payload)
    assert t.meta["vendor"] == "speechmatics"
    assert t.turns[0].text == "Hello."
    assert [x.speaker for x in t.turns] == ["S1", "S2"]


def test_generic_list_of_turn_objects():
    payload = json.dumps([
        {"speaker": "Alice", "text": "hello", "start": 0.0, "end": 1.0},
        {"speaker": "Bob", "text": "hi", "start": 1.0, "end": 2.0}])
    t = parse_asr_json(payload)
    assert t.speakers == ["Alice", "Bob"]
    assert t.turns[0].end_ms == 1000


def test_generic_guesses_millisecond_units_from_magnitude():
    payload = json.dumps([
        {"speaker": "A", "text": "hello", "start": 120000, "end": 124000},
        {"speaker": "B", "text": "hi", "start": 124000, "end": 126000}])
    t = parse_asr_json(payload)
    assert t.turns[0].start_ms == 120000, "large integers are ms, not seconds"


def test_unrecognised_json_raises():
    with pytest.raises(ValueError):
        parse_asr_json(json.dumps({"totally": "unrelated"}))


def test_json_is_never_mistaken_for_plain_text():
    from turnchunk import detect_format
    assert detect_format(open(FIX + "whisperx_diarized.json").read()) == "json"


# ------------------------------------------------------- cloud providers ----
#
# Every major cloud STT encodes time differently, and each is a distinct way to
# be silently wrong: AWS writes seconds as strings, Google appends an "s"
# suffix, Azure counts 100-nanosecond ticks. The text survives a mistake here;
# the citations do not.


def test_aws_transcribe_joins_speaker_labels_to_items():
    """AWS keeps diarization in a separate segment list, addressed by time."""
    t = parse_file(FIX + "aws_transcribe.json")
    assert t.format == "json:aws"
    assert t.speakers == ["spk_0", "spk_1"]
    assert t.turns[0].text == "Hello there."
    assert t.turns[1].text == "How are you?"
    # "0.5" is a string in the source; it must still become 500 ms.
    assert t.turns[0].start_ms == 500 and t.turns[0].end_ms == 1400


def test_aws_punctuation_attaches_without_its_own_speaker():
    """Punctuation items carry no speaker label and must not split a turn."""
    t = parse_file(FIX + "aws_transcribe.json")
    assert len(t) == 2, "punctuation should not create extra turns"
    assert t.turns[0].text.endswith(".")


def test_google_stt_parses_suffixed_times_and_speaker_tags():
    t = parse_file(FIX + "google_stt.json")
    assert t.format == "json:google"
    assert t.turns[0].start_ms == 500, '"0.500s" must parse to 500 ms'
    assert t.turns[1].text == "How are you"
    assert t.speakers == ["SPEAKER_01", "SPEAKER_02"]


def test_google_cumulative_results_are_not_duplicated():
    """Google repeats the whole transcript in its final diarized result."""
    t = parse_file(FIX + "google_stt.json")
    joined = " ".join(x.text for x in t.turns)
    assert joined.count("Hello there") == 1


def test_google_protobuf_duration_objects():
    payload = json.dumps({"results": [{"alternatives": [{"transcript": "hi", "words": [
        {"startTime": {"seconds": 1, "nanos": 500000000}, "endTime": {"seconds": 2},
         "word": "hi", "speakerTag": 1}]}]}]})
    t = parse_asr_json(payload)
    assert t.turns[0].start_ms == 1500


def test_azure_ticks_convert_to_milliseconds():
    """Azure counts 100-nanosecond ticks: 5,000,000 ticks is 500 ms."""
    t = parse_file(FIX + "azure_speech.json")
    assert t.format == "json:azure"
    assert t.turns[0].start_ms == 500
    assert t.turns[0].end_ms == 1400
    assert t.turns[0].text == "Hello there."


def test_google_is_not_mistaken_for_speechmatics():
    """Both use results[].alternatives; the shape inside must disambiguate."""
    assert parse_file(FIX + "google_stt.json").meta["vendor"] == "google"
    speechmatics = json.dumps({"results": [
        {"start_time": 0.5, "end_time": 0.9, "type": "word",
         "alternatives": [{"content": "Hello", "speaker": "S1"}]}]})
    assert parse_asr_json(speechmatics).meta["vendor"] == "speechmatics"
