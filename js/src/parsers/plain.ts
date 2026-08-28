/**
 * Plain-text transcript parser. Mirrors `turnchunk/parsers/plain.py`.
 *
 * "Plain text" is not one format. It is every convention anyone invented for
 * writing down who said what, and real archives mix several in one folder.
 */

import { Transcript, makeTurn } from "../types.js";
import type { Turn } from "../types.js";
import { mergeTurns, parseTimestamp, splitInlineSpeaker } from "./base.js";
import type { ParseOptions } from "./vtt.js";

const LEADING_TS =
  /^\s*[[(<]?\s*(\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s*[\])>]?\s*[-–—]?\s*([\s\S]*)$/;
// Otter/Descript: a line that is only "Name  12:34"
const NAME_THEN_TS = /^\s*([^\d:][^\n]{0,48}?)\s{1,}[[(]?(\d{1,3}:\d{2}(?::\d{2})?)[\])]?\s*$/;
const MD_BOLD = /\*\*(.+?)\*\*|__(.+?)__/g;
const SPEAKER_WITH_TS =
  /^\s*([^:\n]{1,48}?)\s*[[(]\s*(\d{1,3}:\d{2}(?::\d{2})?)\s*[\])]\s*:\s*([\s\S]*)$/;

function stripMarkdown(line: string): string {
  return line.replace(MD_BOLD, (_m, a: string | undefined, b: string | undefined) => a ?? b ?? "");
}

function isSpeakerName(name: string): boolean {
  return splitInlineSpeaker(`${name}: x`)[0] !== null;
}

export function looksLikePlain(text: string): boolean {
  const sample = text.split("\n").slice(0, 200).filter((l) => l.trim());
  if (!sample.length) return false;
  let hits = 0;
  for (const line of sample) {
    const stripped = stripMarkdown(line);
    if (LEADING_TS.test(stripped) || NAME_THEN_TS.test(stripped)) {
      hits++;
      continue;
    }
    if (splitInlineSpeaker(stripped)[0] !== null) hits++;
  }
  return hits >= Math.max(2, sample.length * 0.15);
}

export function parsePlain(text: string, opts: ParseOptions = {}): Transcript {
  const { source = null, merge = true, maxGapMs = null } = opts;
  const lines = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n").split("\n");

  const cues: Turn[] = [];
  let pendingSpeaker: string | null = null;
  let pendingTs: number | null = null;

  const emit = (body: string, speaker: string | null, ts: number | null) => {
    const t = body.trim();
    if (!t) return;
    cues.push(
      makeTurn({
        text: t, speaker, rawSpeaker: speaker, startMs: ts, endMs: null, index: cues.length,
      }),
    );
  };

  for (const rawLine of lines) {
    let line = stripMarkdown(rawLine).trim();
    if (!line) continue;

    // "Alice Smith  0:12" on its own line -- the next lines are their words.
    const nameThenTs = NAME_THEN_TS.exec(line);
    if (nameThenTs && isSpeakerName(nameThenTs[1]!.trim())) {
      pendingSpeaker = nameThenTs[1]!.trim();
      pendingTs = parseTimestamp(nameThenTs[2]!);
      continue;
    }

    // "Alice (00:12): text"
    const withTs = SPEAKER_WITH_TS.exec(line);
    if (withTs && isSpeakerName(withTs[1]!.trim())) {
      emit(withTs[3]!, withTs[1]!.trim(), parseTimestamp(withTs[2]!));
      pendingSpeaker = withTs[1]!.trim();
      pendingTs = null;
      continue;
    }

    let ts: number | null = null;
    const leading = LEADING_TS.exec(line);
    if (leading && leading[2]) {
      ts = parseTimestamp(leading[1]!);
      line = leading[2]!.trim();
    }

    const [speaker, rest] = splitInlineSpeaker(line);
    if (speaker !== null) {
      emit(rest, speaker, ts ?? pendingTs);
      pendingSpeaker = speaker;
      pendingTs = null;
      continue;
    }

    // No speaker on this line: it continues the current one.
    emit(line, pendingSpeaker, ts ?? pendingTs);
    pendingTs = null;
  }

  const turns = merge ? mergeTurns(cues, { maxGapMs }) : cues;
  return new Transcript(turns, source, "plain", {}, { cue_count: cues.length });
}
