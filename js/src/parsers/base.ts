/**
 * Shared parsing helpers. Mirrors `turnchunk/parsers/base.py`.
 *
 * Everything here exists because some vendor's export broke a naive
 * implementation.
 */

import { makeTurn } from "../types.js";
import type { Turn } from "../types.js";

const TS_RE = /^(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?$/;

// WebVTT/SRT inline markup, plus YouTube's per-word timestamps.
const TAG = /<\/?[a-zA-Z][^>]*>|<\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}>/g;

// Caption formatting tags only, applied after entity decoding. A closed list,
// so ordinary angle-bracket text in a transcript is never eaten.
const ESCAPED_TAG = /<\/?(?:i|b|u|c|v|em|strong|ruby|rt|lang|font|br)\b[^>]*\/?>/gi;

const VOICE = /<v(?:\.[^\s>]+)*\s+([^>]+?)\s*>/i;

const INLINE_SPEAKER = /^\s*([^\s:][^:\n]{0,48}?)\s*:\s+(\S[\s\S]*)$/;

const NOT_A_SPEAKER = new Set([
  "note", "notes", "warning", "caution", "example", "tip", "update", "edit",
  "source", "sources", "ref", "reference", "todo", "fixme", "summary",
  "action", "actions", "e.g", "i.e", "http", "https", "subject", "re",
  "from", "to", "cc", "date", "time", "title", "topic", "agenda",
]);
const BAD_IN_NAME = /[,;!?]|\bhttps?\b/i;

const ENTITIES: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
};

/** Decode the HTML entities that actually appear in caption files. */
function unescapeHtml(text: string): string {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (whole, body: string) => {
    if (body[0] === "#") {
      const code =
        body[1] === "x" || body[1] === "X"
          ? parseInt(body.slice(2), 16)
          : parseInt(body.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : whole;
    }
    return ENTITIES[body.toLowerCase()] ?? whole;
  });
}

/** Parse a timestamp to milliseconds, or null if it isn't one. */
export function parseTimestamp(text: string): number | null {
  const m = TS_RE.exec(text.trim());
  if (!m) return null;
  const h = parseInt(m[1] ?? "0", 10);
  const mi = parseInt(m[2]!, 10);
  const s = parseInt(m[3]!, 10);
  const ms = parseInt((m[4] ?? "0").padEnd(3, "0").slice(0, 3), 10);
  return ((h * 60 + mi) * 60 + s) * 1000 + ms;
}

/**
 * Remove caption markup and decode HTML entities.
 *
 * Two passes, because producers disagree about escaping. The WebVTT spec says
 * `&lt;i&gt;` is the literal text `<i>`, but YouTube emits exactly that to mean
 * italics -- so stripping only raw tags leaves `<i>` in the output. Unescaping
 * first is not safe either. So: strip real tags, unescape, then strip again
 * using a closed list, leaving `a < b` and `<div>` untouched.
 */
export function stripTags(text: string): string {
  return unescapeHtml(text.replace(TAG, "")).replace(ESCAPED_TAG, "");
}

/** Pull the speaker out of a `<v Alice>` voice span, if present. */
export function extractVoiceSpeaker(text: string): string | null {
  const m = VOICE.exec(text);
  return m ? m[1]!.trim() : null;
}

/**
 * Is this colon-prefix a speaker label, or just a sentence containing a colon?
 *
 * Conservative on purpose: miss a real label and every turn goes anonymous;
 * accept a false one and "So here's the thing:" becomes a speaker.
 */
export function looksLikeSpeaker(name: string): boolean {
  if (!name || name.length > 48) return false;
  const words = name.split(/\s+/).filter(Boolean);
  if (!words.length || words.length > 6) return false;
  if (BAD_IN_NAME.test(name) || name.endsWith(".")) return false;
  if (!/[\p{L}\p{N}]/u.test(name)) return false;
  if (NOT_A_SPEAKER.has(name.toLowerCase().replace(/^\.+|\.+$/g, ""))) return false;
  if (words.length === 1) return true;
  // Multi-word labels must read as a name: every word capitalised, or a
  // digit/underscore token like "SPEAKER 00".
  return words.every((w) => {
    const c = w[0]!;
    return /\p{Lu}/u.test(c) || /[0-9_]/.test(c);
  });
}

/** Split `"Alice: hello"` into `["Alice", "hello"]`. */
export function splitInlineSpeaker(text: string): [string | null, string] {
  const m = INLINE_SPEAKER.exec(text);
  if (!m) return [null, text];
  const name = m[1]!.trim();
  if (!looksLikeSpeaker(name)) return [null, text];
  return [name, m[2]!.trim()];
}

/**
 * Remove the repeated prefix produced by rolling captions.
 *
 * YouTube auto-captions scroll: each cue repeats the tail of the previous one
 * and appends a new line. Concatenating naively makes almost every sentence
 * appear two or three times, quietly doubling the index.
 */
export function dropRollingDuplicates(prevLines: string[], lines: string[]): string[] {
  if (!prevLines.length || !lines.length) return lines;
  const maxK = Math.min(prevLines.length, lines.length);
  for (let k = maxK; k >= 1; k--) {
    const tail = prevLines.slice(prevLines.length - k);
    const head = lines.slice(0, k);
    if (tail.every((v, i) => v === head[i])) return lines.slice(k);
  }
  return lines;
}

/**
 * Merge consecutive cues by the same speaker into single turns.
 *
 * Subtitle formats emit a cue every few seconds; a *turn* is everything one
 * person said before someone else spoke. Text is accumulated in an array and
 * joined once -- appending to the string on every cue is quadratic.
 */
export function mergeTurns(
  cues: Turn[],
  opts: { maxGapMs?: number | null; join?: string } = {},
): Turn[] {
  const { maxGapMs = null, join = " " } = opts;
  const out: Turn[] = [];
  const parts: string[][] = [];

  for (const cue of cues) {
    const text = cue.text.trim();
    if (!text) continue;
    const prev = out[out.length - 1];
    if (prev && prev.speaker === cue.speaker) {
      let gapOk = true;
      if (maxGapMs !== null && prev.endMs !== null && cue.startMs !== null) {
        gapOk = cue.startMs - prev.endMs <= maxGapMs;
      }
      if (gapOk) {
        parts[parts.length - 1]!.push(text);
        if (cue.endMs !== null) prev.endMs = cue.endMs;
        if (prev.startMs === null && cue.startMs !== null) prev.startMs = cue.startMs;
        continue;
      }
    }
    out.push(
      makeTurn({
        text: "",
        speaker: cue.speaker,
        startMs: cue.startMs,
        endMs: cue.endMs,
        index: out.length,
        rawSpeaker: cue.rawSpeaker,
        meta: { ...cue.meta },
      }),
    );
    parts.push([text]);
  }

  out.forEach((turn, i) => {
    const chunks = parts[i]!;
    turn.text = chunks.length > 1 ? chunks.join(join) : chunks[0]!;
    turn.index = i;
  });
  return out;
}
