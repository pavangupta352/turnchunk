"""Command line interface.

    turnchunk viz meeting.vtt --compare     # see what a naive splitter does
    turnchunk chunk meeting.vtt --json
    turnchunk stats meeting.vtt
    turnchunk speakers meeting.vtt
    turnchunk report transcripts/*.vtt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence

from . import __version__
from .chunker import DEFAULT_TEMPLATE, chunk, render_turn
from .diagnostics import diarization_warnings
from .naive import recursive_split
from .parsers import FORMATS, UnknownFormatError, detect_format, parse_file
from .render import format_timestamp, to_context
from .speakers import resolve_speakers
from .types import Transcript

ESC = chr(27)

# ------------------------------------------------------------------ style ---


class Style:
    """ANSI styling that disables itself when output is piped or NO_COLOR is set."""

    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = (
                sys.stdout.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )
        self.on = enabled

    def _w(self, code: str, text: str) -> str:
        if not self.on:
            return text
        return "{e}[{c}m{t}{e}[0m".format(e=ESC, c=code, t=text)

    def red(self, t):
        return self._w("31", t)

    def green(self, t):
        return self._w("32", t)

    def yellow(self, t):
        return self._w("33", t)

    def blue(self, t):
        return self._w("34", t)

    def dim(self, t):
        return self._w("2", t)

    def bold(self, t):
        return self._w("1", t)


# ------------------------------------------------------------------ utils ---


def _expand(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        out.extend(matches if matches else [p])
    return out


def _load(path: str, args) -> Transcript:
    t = parse_file(
        path,
        format=getattr(args, "format", None),
        max_gap_ms=getattr(args, "max_gap", None),
    )
    if getattr(args, "resolve_speakers", True):
        resolve_speakers(t.turns)
    return t


def _wrap(text: str, width: int) -> List[str]:
    words, lines, buf = text.split(" "), [], ""
    for w in words:
        candidate = "{} {}".format(buf, w) if buf else w
        if len(candidate) > width and buf:
            lines.append(buf)
            buf = w
        else:
            buf = candidate
    if buf:
        lines.append(buf)
    return lines or [""]


def _turns_cut(turns, chunk_texts: Sequence[str]) -> int:
    """How many whole turns were split across chunk boundaries?"""
    cut = 0
    for t in turns:
        body = t.text.strip()
        if len(body) < 8:
            continue
        if not any(body in c for c in chunk_texts):
            cut += 1
    return cut


def _attributed_speaker(text: str) -> Optional[str]:
    """Who does this raw chunk of text say is speaking?

    This is the measure that matters. Splitting a long turn is fine and
    turnchunk does it too -- but when a general-purpose splitter cuts a turn,
    the continuation carries no label, so a retrieved chunk cannot be
    attributed to anyone. That is how a RAG answer ends up crediting a
    commitment to the wrong person.
    """
    from .parsers.base import split_inline_speaker

    first = text.strip().split("\n", 1)[0]
    speaker, _ = split_inline_speaker(first)
    return speaker


def _starts_mid_sentence(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if first.isupper() or first.isdigit() or first in "\"'“‘([{":
        return False
    return first.isalpha()


# ------------------------------------------------------------------- viz ----


def cmd_viz(args) -> int:
    s = Style(None if args.color == "auto" else args.color == "always")
    t = _load(args.file, args)
    width = max(48, args.width - 6)

    ours = chunk(t, target=args.target, overlap=args.overlap)

    if args.compare:
        naive_texts = recursive_split(
            "\n".join(render_turn(x, DEFAULT_TEMPLATE) for x in t.turns),
            target=args.target,
            overlap=args.overlap,
        )
        print()
        print(
            s.bold(s.red("  RecursiveCharacterTextSplitter"))
            + s.dim("  (size={})".format(args.target))
        )
        print(s.dim("  " + "─" * width))
        orphans = 0
        for i, text in enumerate(naive_texts, 1):
            who = _attributed_speaker(text)
            if who is None:
                orphans += 1
            label = who if who else "?"
            head = "  ┌─ chunk {} · {} ".format(i, label)
            marker = "" if who else s.red("  who said this?")
            print(s.dim(head + "─" * max(0, width - len(head))) + marker)
            for line in text.split("\n"):
                for out in _wrap(line, width - 4):
                    print(s.dim("  │ ") + (out if who else s.red(out)))
            print(s.dim("  └" + "─" * (width - 1)))
        print()
        print(
            "  "
            + s.red(
                "{} chunks · {} cannot be attributed to a speaker · "
                "0 carry a timestamp".format(len(naive_texts), orphans)
            )
        )
        print()

    print(s.bold(s.green("  turnchunk")) + s.dim("  (target={})".format(args.target)))
    print(s.dim("  " + "─" * width))
    for i, c in enumerate(ours, 1):
        span = "{}–{}".format(
            format_timestamp(c.start_ms), format_timestamp(c.end_ms)
        )
        who = c.primary_speaker or "unknown"
        head = "  ┌─ chunk {} · {} · {} ".format(i, who, span)
        print(s.dim(head + "─" * max(0, width - len(head))))
        for j, turn in enumerate(c.turns):
            tag = s.yellow("  [overlap]") if j in c.overlap_indices else ""
            line = "{}: {}".format(turn.speaker or "UNKNOWN", turn.text)
            for k, out in enumerate(_wrap(line, width - 4)):
                print(s.dim("  │ ") + out + (tag if k == 0 else ""))
        print(s.dim("  └" + "─" * (width - 1)))
    orphans = sum(1 for c in ours if not c.speakers)
    timed = sum(1 for c in ours if c.start_ms is not None)
    print()
    print(
        "  "
        + s.green(
            "{} chunks · {} cannot be attributed to a speaker · "
            "{} carry a timestamp".format(len(ours), orphans, timed)
        )
    )
    print()
    return 0


# ----------------------------------------------------------------- chunk ----


def cmd_chunk(args) -> int:
    t = _load(args.file, args)
    chunks = chunk(
        t,
        target=args.target,
        overlap=args.overlap,
        min_tail_ratio=args.min_tail_ratio,
    )
    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False))
    elif args.context:
        for c in chunks:
            print(to_context(c))
            print()
    else:
        for i, c in enumerate(chunks, 1):
            print(
                "--- chunk {}/{} · {} · {}–{} · {} chars · {}".format(
                    i,
                    len(chunks),
                    c.primary_speaker,
                    format_timestamp(c.start_ms),
                    format_timestamp(c.end_ms),
                    len(c.text),
                    c.id,
                )
            )
            print(c.text)
            print()
    return 0


# ----------------------------------------------------------------- stats ----


def cmd_stats(args) -> int:
    s = Style(None if args.color == "auto" else args.color == "always")
    t = _load(args.file, args)
    chunks = chunk(t, target=args.target, overlap=args.overlap)

    talk = {}
    for turn in t.turns:
        if turn.speaker:
            talk[turn.speaker] = talk.get(turn.speaker, 0) + len(turn.text)
    total = sum(talk.values()) or 1

    print()
    print("  " + s.bold(os.path.basename(args.file)))
    print("  format          {}".format(t.format))
    print("  turns           {}".format(len(t)))
    print("  speakers        {}".format(len(t.speakers)))
    print(
        "  duration        {}".format(
            format_timestamp(t.duration_ms) if t.has_timestamps else "unknown"
        )
    )
    print("  timestamps      {}".format("yes" if t.has_timestamps else "no"))
    print("  characters      {:,}".format(sum(len(x.text) for x in t.turns)))
    print("  chunks @{:<7} {}".format(args.target, len(chunks)))
    if t.meta.get("rolling_duplicate_cues"):
        print(
            "  "
            + s.yellow("rolling dupes")
            + "   {} cues deduplicated".format(t.meta["rolling_duplicate_cues"])
        )
    print()
    print("  " + s.bold("talk time"))
    for name, chars in sorted(talk.items(), key=lambda kv: -kv[1]):
        pct = chars / total
        bar = "█" * max(1, int(pct * 32))
        print("   {:<22} {} {:5.1%}".format(name[:22], s.blue(bar.ljust(32)), pct))
    print()
    return 0


# -------------------------------------------------------------- speakers ----


def cmd_speakers(args) -> int:
    t = parse_file(args.file, format=args.format)
    _, mapping = resolve_speakers(t.turns, merge_partial_names=not args.no_merge)
    resolved = {}
    for raw, name in mapping.items():
        resolved.setdefault(name, []).append(raw)
    print()
    for name, raws in sorted(resolved.items()):
        variants = [r for r in raws if r != name]
        extra = "  ←  {}".format(", ".join(variants)) if variants else ""
        print("  {}{}".format(name, extra))
    print(
        "\n  {} distinct speakers from {} labels\n".format(len(resolved), len(mapping))
    )
    return 0


# ---------------------------------------------------------------- report ----


def cmd_report(args) -> int:
    s = Style(None if args.color == "auto" else args.color == "always")
    files = _expand(args.files)
    rows, problems = [], 0
    for path in files:
        try:
            t = _load(path, args)
        except (UnknownFormatError, OSError, ValueError) as e:
            rows.append((os.path.basename(path), "unreadable", "-", "-", "-", str(e)[:24]))
            problems += 1
            continue
        chunks = chunk(t, target=args.target, overlap=args.overlap)
        stubs = sum(1 for c in chunks if len(c.text) < args.target * args.min_tail_ratio)
        mid = sum(1 for c in chunks if _starts_mid_sentence(c.text))
        cut = _turns_cut(t.turns, [c.text for c in chunks])
        problems += stubs + mid + cut
        rows.append(
            (
                os.path.basename(path),
                t.format,
                str(len(t)),
                str(len(chunks)),
                str(stubs),
                "{}/{}".format(mid, cut),
            )
        )

    w = max([len(r[0]) for r in rows] + [8])
    print()
    print(
        "  {:<{w}}  {:<16} {:>6} {:>7} {:>6} {:>8}".format(
            "file", "format", "turns", "chunks", "stubs", "mid/cut", w=w
        )
    )
    print(
        "  {}  {} {} {} {} {}".format(
            "-" * w, "-" * 16, "-" * 6, "-" * 7, "-" * 6, "-" * 8
        )
    )
    for r in rows:
        line = "  {:<{w}}  {:<16} {:>6} {:>7} {:>6} {:>8}".format(*r, w=w)
        bad = r[4] not in ("0", "-") or r[1] == "unreadable"
        print(s.red(line) if bad else line)
    print("\n  {} file(s), {} issue(s)\n".format(len(files), problems))
    return 1 if (problems and args.fail_on_issues) else 0


# ------------------------------------------------------------------ lint ----

_LEVEL = {"high": 3, "medium": 2, "low": 1}


def cmd_lint(args) -> int:
    """Flag turn boundaries that look like diarizer mistakes."""
    s = Style(None if args.color == "auto" else args.color == "always")
    t = _load(args.file, args)
    findings = diarization_warnings(
        t.turns,
        max_gap_ms=args.max_gap_ms,
        flap_max_words=args.flap_max_words,
        flap_min_run=args.flap_min_run,
    )
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False))
    else:
        print()
        if not findings:
            print("  " + s.green("no diarization warnings") + s.dim(
                "  (nothing looked wrong; that is not the same as the labels being right)"))
        colour = {"high": s.red, "medium": s.yellow, "low": s.dim}
        for f in findings:
            where = "turn {}".format(f.start_index) if f.start_index == f.end_index \
                else "turns {}-{}".format(f.start_index, f.end_index)
            at = " @" + format_timestamp(f.start_ms) if f.start_ms is not None else ""
            print("  {} {:<20} {}{}".format(
                colour[f.confidence]("[{}]".format(f.confidence).ljust(8)),
                f.kind, s.dim(where + at), ""))
            for line in _wrap(f.reason, 84):
                print("           " + line)
        print()
        print("  {} finding(s)  ".format(len(findings)) + s.dim(
            "heuristics only -- labels are never changed; raw_speaker keeps the original"))
        print()
    if args.fail_on and any(_LEVEL[f.confidence] >= _LEVEL[args.fail_on] for f in findings):
        return 1
    return 0


# --------------------------------------------------------------- formats ----


def cmd_formats(args) -> int:
    print()
    print("  turnchunk detects these from file content, never the extension:\n")
    rows = [
        ("vtt", "WebVTT: Teams <v> spans, Zoom inline names, YouTube rolling captions"),
        ("srt", "SubRip subtitles"),
        ("plain", "[00:12] Alice: ...  |  Alice: ...  |  **Alice:** ...  |  Otter"),
        ("json:whisper", "Whisper / faster-whisper / WhisperX / OpenAI verbose_json"),
        ("json:deepgram", "Deepgram utterances, paragraphs or words"),
        ("json:assemblyai", "AssemblyAI utterances or words (milliseconds)"),
        ("json:rev", "Rev.ai monologues"),
        ("json:speechmatics", "Speechmatics results"),
        ("json:aws", "AWS Transcribe items + speaker_labels (times as strings)"),
        ("json:google", "Google Cloud STT words with '1.500s' times and speakerTag"),
        ("json:azure", "Azure Speech batch phrases (100-nanosecond ticks)"),
        ("json:generic", "Any list of {speaker, text, start, end} objects"),
    ]
    for name, detail in rows:
        print("    {:<20} {}".format(name, detail))
    print()
    return 0


def cmd_detect(args) -> int:
    for path in _expand(args.files):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                fmt = detect_format(fh.read())
        except OSError as e:
            fmt = "error: {}".format(e)
        print("{}\t{}".format(path, fmt or "unknown"))
    return 0


# ------------------------------------------------------------------ main ----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turnchunk",
        description="Chunking that understands who is speaking.",
    )
    p.add_argument(
        "--version", action="version", version="turnchunk {}".format(__version__)
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, with_target=True):
        sp.add_argument("--format", choices=FORMATS, help="force a parser")
        sp.add_argument(
            "--max-gap",
            type=int,
            metavar="MS",
            help="start a new turn after this much silence",
        )
        sp.add_argument(
            "--no-resolve-speakers",
            dest="resolve_speakers",
            action="store_false",
            help="keep raw speaker labels",
        )
        sp.add_argument("--color", choices=("auto", "always", "never"), default="auto")
        if with_target:
            sp.add_argument("-t", "--target", type=int, default=2000)
            sp.add_argument("-o", "--overlap", type=int, default=0)

    sp = sub.add_parser("viz", help="show chunk boundaries (add --compare)")
    sp.add_argument("file")
    sp.add_argument(
        "--compare",
        action="store_true",
        help="also show what a recursive character splitter does",
    )
    sp.add_argument("--width", type=int, default=88)
    common(sp)
    sp.set_defaults(func=cmd_viz, target=600)

    sp = sub.add_parser("chunk", help="chunk a transcript")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--context", action="store_true", help="render for an LLM prompt")
    sp.add_argument("--min-tail-ratio", type=float, default=1 / 3)
    common(sp)
    sp.set_defaults(func=cmd_chunk)

    sp = sub.add_parser("stats", help="speakers, talk time, duration")
    sp.add_argument("file")
    common(sp)
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("speakers", help="show resolved speaker identities")
    sp.add_argument("file")
    sp.add_argument(
        "--no-merge",
        action="store_true",
        help="do not merge partial names into full ones",
    )
    sp.add_argument("--format", choices=FORMATS)
    sp.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    sp.set_defaults(func=cmd_speakers)

    sp = sub.add_parser("report", help="chunk quality across a corpus")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--min-tail-ratio", type=float, default=1 / 3)
    sp.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="exit non-zero when problems are found (for CI)",
    )
    common(sp)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("lint", help="flag likely diarizer mistakes (flips, flapping, ghosts)")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--max-gap-ms", type=int, default=300,
                    help="a speaker change with less silence than this, mid-sentence, is a flip")
    sp.add_argument("--flap-max-words", type=int, default=4)
    sp.add_argument("--flap-min-run", type=int, default=4)
    sp.add_argument("--fail-on", choices=("high", "medium", "low"),
                    help="exit non-zero if any finding is at or above this confidence")
    common(sp, with_target=False)
    sp.set_defaults(func=cmd_lint)

    sp = sub.add_parser("detect", help="print the detected format for each file")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("formats", help="list supported transcript formats")
    sp.set_defaults(func=cmd_formats)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UnknownFormatError as e:
        print("turnchunk: {}".format(e), file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print("turnchunk: {}".format(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
