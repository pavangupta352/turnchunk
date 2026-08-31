# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-08-31

Found by attacking the published library with hostile input. None of these
produce a wrong answer, which is why the correctness suite never caught them —
they produce a wrong *capability*.

### Security

- **`parse()` no longer reads the filesystem.** It previously opened a file
  whenever the string it was given happened to be a valid path. Applications
  routinely call `parse(request.body)` on user input, which turned that into
  arbitrary local file disclosure: a config or log file that happened to parse
  came back as "speaker turns". A `str` is now always transcript content.
- **Bounded speaker resolution.** `build_speaker_map()` compared every label
  against every other, and the plain-text parser treats any `Word: text` line
  as a speaker — so a 6,000-line uploaded log manufactured 6,000 fake speakers
  and cost **13.7 seconds of CPU per request**. Candidates are now found through
  a token index, and partial-name merging is skipped above
  `MAX_MERGE_LABELS` (1000) distinct labels. Same input now takes 30ms.

### Breaking

- `parse("meeting.vtt")` no longer reads that file. Use `parse_file()`, or pass
  a `pathlib.Path`. The error message says so explicitly. `parse(text)` is
  unchanged.
- The adapters follow the same rule and gained explicit file variants:
  `TurnChunkSplitter.split_transcript_file()`,
  `TurnChunkNodeParser.parse_transcript_file()`,
  `ConversationChunker.chunk_file()`.

### Fixed

- `NaN` and `Infinity` in vendor JSON raised `OverflowError` from `round()`.
  Python's `json` module accepts both, and an unexpected exception type from a
  parser is worse than a wrong number because callers cannot defend against it.
  Every time converter now returns `None` for non-finite values, in both
  languages.

## [0.2.0] — 2026-08-28

### Added
- **AWS Transcribe**, **Google Cloud Speech-to-Text** and **Azure Speech**
  parsers, closing the gap between the "reads anything" claim and the three
  largest cloud providers being absent. Each encodes time differently and each
  is a distinct way to be silently wrong: AWS writes seconds as strings, Google
  appends an `"s"` suffix, Azure counts 100-nanosecond ticks.
- Handling for AWS's separate `speaker_labels` segment list, addressed by time
  range, and for its punctuation items which carry no speaker of their own.
- De-duplication of Google's cumulative diarized results, which otherwise repeat
  the entire transcript in the final result.

### Fixed
- Google Cloud STT output was being claimed by the Speechmatics detector, since
  both use `results[].alternatives`. Detection now discriminates on the shape
  *inside* an alternative.

### Changed
- README badges: the PyPI ones rendered as "package or version not found"
  because GitHub's camo proxy had cached them from before publication. Added a
  CI status badge at the same time.

## [0.1.0] — 2026-08-28

First release.

### Added
- `parse()` with content-based format detection across WebVTT, SubRip, plain
  text, and the JSON emitted by Whisper/WhisperX, Deepgram, AssemblyAI, Rev.ai
  and Speechmatics, plus a generic fallback.
- Handling for YouTube rolling captions, Teams `<v>` voice spans, Zoom inline
  speaker labels, per-word timestamps, and AssemblyAI's millisecond units.
- `chunk()` implementing the four rules: boundaries only at turn boundaries,
  sentence-level splitting of oversized turns with the speaker preserved,
  whole-turn overlap that is explicitly marked, and backwards tail merging.
- `resolve_speakers()` identity resolution that refuses ambiguous merges.
- Dependency-free sentence segmentation, including CJK and uncapitalised ASR
  output.
- `to_context()` prompt rendering.
- CLI: `viz`, `chunk`, `stats`, `speakers`, `report`, `detect`, `formats`.
- Adapters for LangChain, LlamaIndex and chonkie.
- A TypeScript port published to npm, verified byte-identical to the Python
  package against a shared conformance corpus — including chunk ids. Runs on
  Node 18+, browsers, Deno, Bun and edge runtimes with zero dependencies.
- `scripts/verify-prior-art.sh`, which reproduces every prior-art claim in the
  README as live queries.

### Notes
- Validated against real YouTube caption exports rather than only hand-written
  fixtures, which surfaced three bugs: HTML-escaped formatting tags leaking into
  chunk text, and two quadratic blowups (one in sentence segmentation, one in
  cue merging). A 1.7M-character transcript now parses in 450ms and chunks in
  16ms.
