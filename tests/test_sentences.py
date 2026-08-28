"""Sentence segmentation tests.

Splitting is only ever used to break a turn that is too long to fit, so the
priority is: never lose a character, never split inside an abbreviation, and
still work on scripts and registers that have no capital letters.
"""

from __future__ import annotations

import pytest

from turnchunk import split_sentences


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello there. How are you? I'm fine!", 3),
        ("Dr. Smith said the U.S. market grew 3.14 percent. Then he left.", 2),
        ("We met with J. Smith on Feb. 3. It went well.", 2),
        ("Visit example.com and click through. Then you're done.", 2),
        ("The rate is 3.5 percent annually. That is fixed.", 2),
        ("One sentence only", 1),
        ("", 0),
    ],
)
def test_segmentation(text, expected):
    assert len(split_sentences(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hello there. How are you? I'm fine!",
        "Dr. Smith said the U.S. market grew 3.14 percent.",
        "そうですね。次の議題に移りましょう。",
        "no caps at all here. it just keeps going. like asr output does",
    ],
)
def test_never_loses_a_character(text):
    assert "".join(split_sentences(text)) == text


def test_cjk_splits_without_whitespace():
    """CJK has no space after 。 -- requiring one means never splitting at all."""
    assert len(split_sentences("そうですね。次の議題に移りましょう。それでいいですか。")) == 3


def test_uncapitalised_asr_output_still_splits():
    text = (
        "so i talked to the vendor yesterday about the renewal and they said "
        "the price would go up. i pushed back on that pretty hard because we "
        "had an agreement. anyway they are checking with their manager now"
    )
    assert len(split_sentences(text)) == 3


def test_capitalised_text_stays_conservative():
    """With capitals present, a lowercase continuation is not a boundary."""
    assert len(split_sentences("The item costs 5 dollars. The other one is free.")) == 2
