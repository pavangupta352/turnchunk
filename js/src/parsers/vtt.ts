/**
 * WebVTT parser. Mirrors `turnchunk/parsers/vtt.py`.
 *
 * Teams wraps the speaker in `<v Alice>`; Zoom writes `Alice: text` inside the
 * cue; YouTube auto-captions scroll and carry per-word timestamps. All valid
 * WebVTT, all different.
 */

import { Transcript, makeTurn } from "../types.js";
import type { Turn } from "../types.js";
import {
  dropRollingDuplicates,
  extractVoiceSpeaker,
  mergeTurns,
  parseTimestamp,
  splitInlineSpeaker,
  stripTags,
} from "./base.js";

const BLOCK_KEYWORDS = ["NOTE", "STYLE", "REGION", "COMMENT"];

export interface ParseOptions {
  source?: string | null;
  merge?: boolean;
  maxGapMs?: number | null;
}

export function looksLikeVtt(text: string): boolean {
  const head = text.replace(/^[﻿\s]+/, "").slice(0, 64).toUpperCase();
  if (head.startsWith("WEBVTT")) return true;
  // Headerless VTT still uses "." for fractional seconds; SubRip uses ",".
  // Requiring the period is what keeps a .srt from being claimed here.
  return /\d{2}:\d{2}\.\d{3}\s*-->/.test(text.slice(0, 4000));
}

function parseCueTiming(line: string): [number | null, number | null] {
  const parts = line.split(/\s*-->\s*/);
  if (parts.length !== 2) return [null, null];
  const left = parts[0]!.trim();
  const start = left ? parseTimestamp(left.split(/\s+/).pop()!) : null;
  const right = parts[1]!.trim();
  // The end timestamp may be followed by cue settings: "align:start position:0%".
  const end = right ? parseTimestamp(right.split(/\s+/)[0]!) : null;
  return [start, end];
}

export function parseVtt(text: string, opts: ParseOptions = {}): Transcript {
  const { source = null, merge = true, maxGapMs = null } = opts;
  const lines = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n").split("\n");

  const cues: Turn[] = [];
  let prevLines: string[] = [];
  let rollingHits = 0;
  let i = 0;

  while (i < lines.length) {
    const line = lines[i]!.trim();

    if (!line) { i++; continue; }
    if (line.toUpperCase().startsWith("WEBVTT")) { i++; continue; }
    if (BLOCK_KEYWORDS.some((k) => line.startsWith(k))) {
      while (i < lines.length && lines[i]!.trim()) i++;
      continue;
    }

    // A cue is: [identifier] / timestamp line / payload lines.
    let tsLine = line;
    if (!tsLine.includes("-->")) {
      if (i + 1 < lines.length && lines[i + 1]!.includes("-->")) {
        i++;
        tsLine = lines[i]!.trim();
      } else {
        i++;
        continue;
      }
    }

    const [startMs, endMs] = parseCueTiming(tsLine);
    i++;

    const payload: string[] = [];
    while (i < lines.length && lines[i]!.trim()) {
      payload.push(lines[i]!);
      i++;
    }
    if (!payload.length) continue;

    let speaker = extractVoiceSpeaker(payload.join("\n"));
    const cleaned = payload.map((p) => stripTags(p).trim()).filter(Boolean);
    if (!cleaned.length) continue;

    const deduped = dropRollingDuplicates(prevLines, cleaned);
    if (deduped.length !== cleaned.length) rollingHits++;
    prevLines = cleaned;
    if (!deduped.length) continue;

    let body = deduped.join(" ").trim();
    if (speaker === null) {
      const [inline, rest] = splitInlineSpeaker(body);
      if (inline !== null) {
        speaker = inline;
        body = rest;
      }
    }
    if (!body) continue;

    cues.push(
      makeTurn({
        text: body,
        speaker,
        rawSpeaker: speaker,
        startMs,
        endMs,
        index: cues.length,
      }),
    );
  }

  const turns = merge ? mergeTurns(cues, { maxGapMs }) : cues;
  return new Transcript(turns, source, "vtt", {}, {
    cue_count: cues.length,
    rolling_duplicate_cues: rollingHits,
  });
}
