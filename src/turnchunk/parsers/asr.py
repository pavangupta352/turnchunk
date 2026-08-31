"""Speech-to-text vendor JSON.

Every ASR vendor ships a different JSON shape, and the differences are not
cosmetic. The one that silently ruins a pipeline is **time units**: Whisper,
Deepgram, Rev and Speechmatics emit seconds as floats, AssemblyAI emits integer
milliseconds. Read an AssemblyAI file as seconds and every timestamp is off by a
factor of a thousand -- the transcript still "works", the citations just point
at the wrong moment forever.

Supported, detected from the payload rather than the filename:

* Whisper / faster-whisper / OpenAI ``verbose_json`` (``segments[]``)
* WhisperX and whisper.cpp diarized output (``segments[].speaker``)
* Deepgram (``results.utterances`` / ``paragraphs`` / ``words``)
* AssemblyAI (``utterances[]`` / ``words[]``, milliseconds)
* Rev.ai (``monologues[].elements[]``)
* Speechmatics (``results[]`` with ``alternatives[].speaker``)
* AWS Transcribe (``results.items`` + ``results.speaker_labels``, times as strings)
* Google Cloud STT (``results[].alternatives[].words`` with ``"1.500s"`` times)
* Azure Speech batch (``recognizedPhrases`` with 100-nanosecond ticks)
* A generic fallback for any list of ``{speaker, text, start, end}`` objects

Each cloud vendor encodes time differently, and every one of them is a way to
be silently wrong: AWS writes seconds as *strings*, Google appends an ``s``
suffix, and Azure counts 100-nanosecond ticks.
"""

from __future__ import annotations

import json as _json
import math
from typing import Any, Dict, List, Optional, Sequence

from ..types import Transcript, Turn
from .base import merge_turns


def looks_like_asr_json(text: str) -> bool:
    stripped = text.lstrip("﻿ \t\r\n")
    if stripped[:1] not in ("{", "["):
        return False
    try:
        data = _json.loads(stripped)
    except ValueError:
        return False
    return _identify(data) is not None


def _identify(data: Any) -> Optional[str]:
    if isinstance(data, list):
        return "generic_list" if _is_turn_list(data) else None
    if not isinstance(data, dict):
        return None
    if "monologues" in data:
        return "rev"
    if "recognizedPhrases" in data or "combinedRecognizedPhrases" in data:
        return "azure"
    results = data.get("results")
    if isinstance(results, dict) and "channels" in results:
        return "deepgram"
    if isinstance(results, dict) and ("speaker_labels" in results or "items" in results):
        return "aws"
    if isinstance(results, list) and results and isinstance(results[0], dict):
        # Google and Speechmatics both use results[].alternatives, so the
        # discriminator is what is *inside* an alternative: Google carries a
        # transcript and word list, Speechmatics a single token's content.
        first = results[0]
        alts = first.get("alternatives")
        if isinstance(alts, list) and alts and isinstance(alts[0], dict):
            alt = alts[0]
            if "words" in alt or "transcript" in alt:
                return "google"
            if "content" in alt or "start_time" in first:
                return "speechmatics"
    if "utterances" in data and isinstance(data["utterances"], list):
        # AssemblyAI has a top-level utterances[] alongside words[] in ms.
        return "assemblyai" if "words" in data or "id" in data else "generic_dict"
    if "segments" in data and isinstance(data["segments"], list):
        return "whisper"
    for key in ("turns", "transcript", "items", "entries"):
        if isinstance(data.get(key), list) and _is_turn_list(data[key]):
            return "generic_dict"
    return None


def _is_turn_list(items: Sequence[Any]) -> bool:
    if not items or not isinstance(items[0], dict):
        return False
    keys = set(items[0])
    return bool(keys & {"text", "transcript", "value", "content"})


# --------------------------------------------------------------- helpers ----

def _sec_to_ms(v: Any) -> Optional[int]:
    """Seconds to milliseconds. NaN and infinity become None, not a crash.

    Python's json module accepts the non-standard NaN/Infinity literals, so a
    vendor file containing them would otherwise reach round() and raise
    OverflowError -- an exception type callers do not expect from a parser.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f * 1000) if math.isfinite(f) else None


def _ms(v: Any) -> Optional[int]:
    """Milliseconds through, with the same NaN/infinity guard as _sec_to_ms."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f) if math.isfinite(f) else None


def _speaker(value: Any, prefix: str = "SPEAKER") -> Optional[str]:
    """Normalise a speaker field. Integer ids become ``SPEAKER_00``-style labels."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{prefix}_{value:02d}"
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"{prefix}_{int(text):02d}"
    return text


# --------------------------------------------------------------- parsers ----

def _from_whisper(data: Dict[str, Any]) -> List[Turn]:
    out: List[Turn] = []
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append(Turn(
            text=text,
            speaker=_speaker(seg.get("speaker")),
            start_ms=_sec_to_ms(seg.get("start")),
            end_ms=_sec_to_ms(seg.get("end")),
            index=len(out),
        ))
    return out


def _from_assemblyai(data: Dict[str, Any]) -> List[Turn]:
    """AssemblyAI. Times are integer milliseconds, not seconds."""
    out: List[Turn] = []
    utterances = data.get("utterances") or []
    if utterances:
        for u in utterances:
            text = (u.get("text") or "").strip()
            if not text:
                continue
            out.append(Turn(
                text=text,
                speaker=_speaker(u.get("speaker")),
                start_ms=_ms(u.get("start")),
                end_ms=_ms(u.get("end")),
                index=len(out),
            ))
        return out
    return _group_words(data.get("words") or [], text_key="text", to_ms=_ms)


def _from_deepgram(data: Dict[str, Any]) -> List[Turn]:
    results = data.get("results") or {}
    utterances = results.get("utterances") or []
    if utterances:
        out: List[Turn] = []
        for u in utterances:
            text = (u.get("transcript") or "").strip()
            if not text:
                continue
            out.append(Turn(
                text=text,
                speaker=_speaker(u.get("speaker")),
                start_ms=_sec_to_ms(u.get("start")),
                end_ms=_sec_to_ms(u.get("end")),
                index=len(out),
            ))
        return out

    channels = results.get("channels") or []
    alt = (channels[0].get("alternatives") or [{}])[0] if channels else {}

    paragraphs = ((alt.get("paragraphs") or {}).get("paragraphs")) or []
    if paragraphs:
        out = []
        for p in paragraphs:
            sentences = p.get("sentences") or []
            text = " ".join((s.get("text") or "").strip() for s in sentences).strip()
            if not text:
                continue
            out.append(Turn(
                text=text,
                speaker=_speaker(p.get("speaker")),
                start_ms=_sec_to_ms(p.get("start") or (sentences[0].get("start") if sentences else None)),
                end_ms=_sec_to_ms(p.get("end") or (sentences[-1].get("end") if sentences else None)),
                index=len(out),
            ))
        return out

    return _group_words(alt.get("words") or [], text_key="punctuated_word",
                        fallback_key="word", to_ms=_sec_to_ms)


def _from_rev(data: Dict[str, Any]) -> List[Turn]:
    out: List[Turn] = []
    for mono in data.get("monologues", []):
        elements = mono.get("elements") or []
        text = "".join(str(e.get("value", "")) for e in elements).strip()
        if not text:
            continue
        times = [e for e in elements if e.get("ts") is not None]
        out.append(Turn(
            text=text,
            speaker=_speaker(mono.get("speaker")),
            start_ms=_sec_to_ms(times[0]["ts"]) if times else None,
            end_ms=_sec_to_ms(times[-1].get("end_ts")) if times else None,
            index=len(out),
        ))
    return out


def _from_speechmatics(data: Dict[str, Any]) -> List[Turn]:
    words = []
    for item in data.get("results", []):
        alts = item.get("alternatives") or [{}]
        words.append({
            "text": alts[0].get("content", ""),
            "speaker": alts[0].get("speaker"),
            "start": item.get("start_time"),
            "end": item.get("end_time"),
            "type": item.get("type"),
        })
    return _group_words(words, text_key="text", to_ms=_sec_to_ms, join_punct=True)


def _from_generic(items: Sequence[Dict[str, Any]]) -> List[Turn]:
    """Best effort over an unknown but turn-shaped list.

    Time units are guessed from magnitude: a 'start' of 125000 in a file whose
    values are integers is milliseconds, not thirty-four hours of audio.
    """
    def pick(d: Dict[str, Any], *keys):
        for k in keys:
            if d.get(k) not in (None, ""):
                return d[k]
        return None

    raw_starts = [pick(d, "start", "start_ms", "startTime", "ts", "begin")
                  for d in items if isinstance(d, dict)]
    numeric = [float(v) for v in raw_starts if isinstance(v, (int, float))]
    use_ms = bool(numeric) and max(numeric) > 10_000 and all(
        float(v).is_integer() for v in numeric
    )
    conv = _ms if use_ms else _sec_to_ms

    out: List[Turn] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        text = str(pick(d, "text", "transcript", "value", "content") or "").strip()
        if not text:
            continue
        out.append(Turn(
            text=text,
            speaker=_speaker(pick(d, "speaker", "speaker_label", "speakerId", "spk")),
            start_ms=conv(pick(d, "start", "start_ms", "startTime", "ts", "begin")),
            end_ms=conv(pick(d, "end", "end_ms", "endTime", "end_ts", "stop")),
            index=len(out),
        ))
    return out


def _group_words(words, *, text_key, to_ms, fallback_key=None, join_punct=False) -> List[Turn]:
    """Collapse word-level output into turns, breaking whenever the speaker changes."""
    out: List[Turn] = []
    current: Optional[Turn] = None
    for w in words:
        if not isinstance(w, dict):
            continue
        token = w.get(text_key) or (w.get(fallback_key) if fallback_key else None) or ""
        token = str(token)
        if not token:
            continue
        spk = _speaker(w.get("speaker"))
        start = to_ms(w.get("start"))
        end = to_ms(w.get("end"))
        is_punct = join_punct and w.get("type") == "punctuation"

        if current is None or (spk != current.speaker and not is_punct):
            current = Turn(text=token, speaker=spk, start_ms=start, end_ms=end,
                           index=len(out))
            out.append(current)
        else:
            sep = "" if is_punct else " "
            current.text = f"{current.text}{sep}{token}"
            if end is not None:
                current.end_ms = end
    for t in out:
        t.text = t.text.strip()
    return [t for t in out if t.text]


def _from_aws(data: Dict[str, Any]) -> List[Turn]:
    """AWS Transcribe.

    Times arrive as *strings* (``"12.34"``), and with diarization enabled the
    speaker lives on ``results.speaker_labels.segments`` rather than on the
    items themselves -- newer output also copies it onto each item, so prefer
    that and fall back to a time-range lookup.
    """
    results = data.get("results") or {}
    items = results.get("items") or []

    segments = ((results.get("speaker_labels") or {}).get("segments")) or []
    ranges = [
        (
            _sec_to_ms(seg.get("start_time")),
            _sec_to_ms(seg.get("end_time")),
            _speaker(seg.get("speaker_label")),
        )
        for seg in segments
    ]

    def speaker_at(start_ms: Optional[int]) -> Optional[str]:
        if start_ms is None:
            return None
        for lo, hi, who in ranges:
            if lo is not None and hi is not None and lo <= start_ms <= hi:
                return who
        return None

    words: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        alts = item.get("alternatives") or [{}]
        content = (alts[0] or {}).get("content", "")
        if not content:
            continue
        start = _sec_to_ms(item.get("start_time"))
        spk = _speaker(item.get("speaker_label")) or speaker_at(start)
        words.append({
            "text": content,
            "speaker": spk,
            "start": item.get("start_time"),
            "end": item.get("end_time"),
            "type": "punctuation" if item.get("type") == "punctuation" else "word",
        })

    # Punctuation items carry no speaker; let them attach to the running turn.
    for i, w in enumerate(words):
        if w["type"] == "punctuation" and i:
            w["speaker"] = words[i - 1]["speaker"]

    return _group_words(words, text_key="text", to_ms=_sec_to_ms, join_punct=True)


_GOOGLE_TIME = "s"


def _google_time(value: Any) -> Optional[int]:
    """Google writes durations as ``"1.500s"``."""
    if value is None:
        return None
    if isinstance(value, dict):  # protobuf {seconds, nanos}
        try:
            secs = float(value.get("seconds", 0))
            nanos = float(value.get("nanos", 0))
        except (TypeError, ValueError):
            return None
        total = secs * 1000 + nanos / 1e6
        return round(total) if math.isfinite(total) else None
    text = str(value).strip()
    if text.endswith(_GOOGLE_TIME):
        text = text[:-1]
    return _sec_to_ms(text)


def _from_google(data: Dict[str, Any]) -> List[Turn]:
    """Google Cloud Speech-to-Text.

    Diarization results are cumulative: the final result repeats every word
    with its ``speakerTag``, so taking the last result with words avoids
    emitting the transcript several times over.
    """
    results = data.get("results") or []
    best_words: List[Dict[str, Any]] = []
    fallback: List[Turn] = []

    for res in results:
        alts = (res or {}).get("alternatives") or [{}]
        alt = alts[0] or {}
        words = alt.get("words") or []
        if words:
            best_words = words  # cumulative; the last one wins
        elif alt.get("transcript"):
            fallback.append(
                Turn(text=str(alt["transcript"]).strip(), index=len(fallback))
            )

    if not best_words:
        return fallback

    normalised = [
        {
            "text": w.get("word", ""),
            "speaker": _speaker(w.get("speakerTag") or w.get("speaker_tag")),
            "start": w.get("startTime") or w.get("start_time"),
            "end": w.get("endTime") or w.get("end_time"),
        }
        for w in best_words
        if isinstance(w, dict)
    ]
    return _group_words(normalised, text_key="text", to_ms=_google_time)


def _ticks_to_ms(offset: Any, duration: Any = None) -> Optional[int]:
    if offset is None:
        return None
    try:
        total = float(offset) + (float(duration) if duration is not None else 0.0)
    except (TypeError, ValueError):
        return None
    return round(total / 10_000) if math.isfinite(total) else None


def _from_azure(data: Dict[str, Any]) -> List[Turn]:
    """Azure Speech batch transcription. Time is in 100-nanosecond ticks."""
    out: List[Turn] = []
    for phrase in data.get("recognizedPhrases") or []:
        if not isinstance(phrase, dict):
            continue
        best = (phrase.get("nBest") or [{}])[0] or {}
        text = str(best.get("display") or best.get("lexical") or "").strip()
        if not text:
            continue
        # Ticks are 100-nanosecond units; guard the same way as everywhere else.
        start_ms = _ticks_to_ms(phrase.get("offsetInTicks"))
        end_ms = _ticks_to_ms(phrase.get("offsetInTicks"), phrase.get("durationInTicks"))
        out.append(Turn(
            text=text,
            speaker=_speaker(phrase.get("speaker")),
            start_ms=start_ms,
            end_ms=end_ms,
            index=len(out),
        ))
    return out


_DISPATCH = {
    "whisper": lambda d: _from_whisper(d),
    "assemblyai": lambda d: _from_assemblyai(d),
    "deepgram": lambda d: _from_deepgram(d),
    "rev": lambda d: _from_rev(d),
    "speechmatics": lambda d: _from_speechmatics(d),
    "aws": lambda d: _from_aws(d),
    "google": lambda d: _from_google(d),
    "azure": lambda d: _from_azure(d),
    "generic_list": lambda d: _from_generic(d),
    "generic_dict": lambda d: _from_generic(
        d.get("utterances") or d.get("turns") or d.get("transcript")
        or d.get("items") or d.get("entries") or []
    ),
}


def parse_asr_json(
    text: str,
    *,
    source: Optional[str] = None,
    merge: bool = True,
    max_gap_ms: Optional[int] = None,
    vendor: Optional[str] = None,
) -> Transcript:
    """Parse speech-to-text vendor JSON into turns."""
    data = _json.loads(text.lstrip("﻿"))
    kind = vendor or _identify(data)
    if kind is None or kind not in _DISPATCH:
        raise ValueError(
            "Unrecognised ASR JSON. Supported: "
            + ", ".join(sorted(_DISPATCH)) + "."
        )
    cues = _DISPATCH[kind](data)
    turns = merge_turns(cues, max_gap_ms=max_gap_ms) if merge else cues
    return Transcript(
        turns=turns,
        source=source,
        format=f"json:{kind}",
        meta={"vendor": kind, "cue_count": len(cues)},
    )
