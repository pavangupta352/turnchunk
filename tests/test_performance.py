"""Performance regressions.

Two quadratic blowups were found by running the parser over real 16 MB YouTube
caption exports, and both looked completely innocent in review:

* ``sentence_spans`` sliced ``text[m.end():]`` inside its boundary loop, copying
  the rest of the document on every one of ~10k boundaries.
* ``merge_turns`` rebuilt ``turn.text`` on every merged cue, so a file whose
  46,958 cues all belong to one speaker copied the accumulated string 46,958
  times.

Neither changed any output, so no correctness test caught them. These do.

The thresholds are deliberately loose -- they exist to catch O(n^2), not to
police milliseconds on a shared CI runner.
"""

from __future__ import annotations

import time

import pytest

from turnchunk import Turn, chunk, parse, split_sentences

SENTENCE = "The migration plan needs about a month of runway, maybe a little more. "


def build_vtt(cues: int, speakers: int = 1) -> str:
    out = ["WEBVTT", ""]
    for i in range(cues):
        start, end = i * 2, i * 2 + 2
        who = f"Speaker {i % speakers}" if speakers > 1 else "Speaker 0"
        out += [
            f"{i // 3600:02d}:{(start // 60) % 60:02d}:{start % 60:02d}.000 --> "
            f"{i // 3600:02d}:{(end // 60) % 60:02d}:{end % 60:02d}.000",
            f"<v {who}>{SENTENCE.strip()}</v>",
            "",
        ]
    return "\n".join(out)


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def growth(small: float, large: float, factor: int) -> float:
    """How much worse than linear was the larger run?"""
    if small <= 0:
        return 1.0
    return (large / small) / factor


# --------------------------------------------------------------------------

def test_sentence_splitting_is_not_quadratic():
    small = SENTENCE * 500
    large = SENTENCE * 4000  # 8x the work
    _, t_small = timed(lambda: split_sentences(small))
    _, t_large = timed(lambda: split_sentences(large))
    assert growth(t_small, t_large, 8) < 3.0, (
        f"sentence splitting scales super-linearly: {t_small:.3f}s -> {t_large:.3f}s"
    )


def test_merging_many_cues_from_one_speaker_is_not_quadratic():
    """The pathological real-world case: one speaker, tens of thousands of cues."""
    _, t_small = timed(lambda: parse(build_vtt(2_000), format="vtt"))
    _, t_large = timed(lambda: parse(build_vtt(16_000), format="vtt"))  # 8x
    assert growth(t_small, t_large, 8) < 3.0, (
        f"cue merging scales super-linearly: {t_small:.3f}s -> {t_large:.3f}s"
    )


def test_chunking_a_very_long_single_turn_is_not_quadratic():
    def run(n):
        turns = [Turn(text=SENTENCE * n, speaker="A", index=0)]
        return chunk(turns, target=2000)

    _, t_small = timed(lambda: run(500))
    _, t_large = timed(lambda: run(4000))  # 8x
    assert growth(t_small, t_large, 8) < 3.0, (
        f"splitting a long turn scales super-linearly: {t_small:.3f}s -> {t_large:.3f}s"
    )


@pytest.mark.parametrize("cues,budget_s", [(10_000, 6.0)])
def test_large_transcript_completes_in_reasonable_time(cues, budget_s):
    """A generous absolute ceiling, well above any healthy machine."""
    vtt = build_vtt(cues, speakers=4)
    transcript, t_parse = timed(lambda: parse(vtt, format="vtt"))
    chunks, t_chunk = timed(lambda: chunk(transcript, target=2000))
    assert chunks
    total = t_parse + t_chunk
    assert total < budget_s, f"{cues} cues took {total:.2f}s (budget {budget_s}s)"
