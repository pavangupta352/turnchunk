# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
