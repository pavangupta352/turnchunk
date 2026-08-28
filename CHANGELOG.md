# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
