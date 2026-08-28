/** SubRip (.srt) parser. Mirrors `turnchunk/parsers/srt.py`. */

import { Transcript, makeTurn } from "../types.js";
import type { Turn } from "../types.js";
import {
  dropRollingDuplicates,
  mergeTurns,
  parseTimestamp,
  splitInlineSpeaker,
  stripTags,
} from "./base.js";
import type { ParseOptions } from "./vtt.js";

export function looksLikeSrt(text: string): boolean {
  return /^\s*\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->/m.test(text.slice(0, 4000));
}

export function parseSrt(text: string, opts: ParseOptions = {}): Transcript {
  const { source = null, merge = true, maxGapMs = null } = opts;
  const blocks = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n").trim().split(/\n\s*\n/);

  const cues: Turn[] = [];
  let prevLines: string[] = [];

  for (const block of blocks) {
    let lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    if (/^\d+$/.test(lines[0]!.trim()) && lines.length > 1) lines = lines.slice(1);
    if (!lines.length || !lines[0]!.includes("-->")) continue;

    const parts = lines[0]!.trim().split(/\s*-->\s*/);
    const startMs = parts[0] ? parseTimestamp(parts[0].trim()) : null;
    const endMs =
      parts.length > 1 && parts[1]!.trim()
        ? parseTimestamp(parts[1]!.trim().split(/\s+/)[0]!)
        : null;

    const cleaned = lines.slice(1).map((l) => stripTags(l).trim()).filter(Boolean);
    if (!cleaned.length) continue;

    const deduped = dropRollingDuplicates(prevLines, cleaned);
    prevLines = cleaned;
    if (!deduped.length) continue;

    let body = deduped.join(" ").trim();
    const [speaker, rest] = splitInlineSpeaker(body);
    if (speaker !== null) body = rest;
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
  return new Transcript(turns, source, "srt", {}, { cue_count: cues.length });
}
