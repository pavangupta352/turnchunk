/**
 * Diarization sanity checks. Mirrors `turnchunk/diagnostics.py`.
 *
 * A perfectly chunked turn can still carry the wrong name, because the
 * diarizer that produced the labels made a mistake upstream. turnchunk cannot
 * fix that -- it needs the audio, or at least a model. What it can do is
 * notice the symptoms, which leave fingerprints in timing and text:
 *
 * - mid-utterance flip: a speaker change with no pause where the sentence
 *   visibly continues across the boundary.
 * - flapping: a run of very short turns bouncing between the same two labels.
 *   Real back-channel looks similar but only one side is short.
 * - ghost speaker: a diarizer-generated label with a handful of words in an
 *   hour of audio. A named speaker who spoke once is never flagged.
 *
 * Heuristics, labelled as such. Nothing is ever changed.
 */

import { isGenericLabel } from "./speakers.js";
import { turnsFrom } from "./types.js";
import type { Turn, TurnsLike } from "./types.js";

const TERMINAL = ".!?…。！？";
const CLOSERS = "\"'’”)]}」』";

export type FindingKind = "mid_utterance_flip" | "flapping" | "ghost_speaker";
export type Confidence = "high" | "medium" | "low";

/** One suspicious region of a transcript. */
export interface Finding {
  kind: FindingKind;
  /** Index of the first turn involved. */
  startIndex: number;
  /** Index of the last turn involved (inclusive). */
  endIndex: number;
  speakers: string[];
  reason: string;
  /** How strongly the evidence points. */
  confidence: Confidence;
  /**
   * Window a consumer should re-examine. For a flip this spans both turns, so
   * a pipeline with the audio can re-diarize exactly the suspect region.
   */
  startMs: number | null;
  endMs: number | null;
}

export interface DiagnosticsOptions {
  maxGapMs?: number;
  flapMaxWords?: number;
  flapMinRun?: number;
  ghostMaxTurns?: number;
  ghostMaxShare?: number;
}

/** Python's repr() for the simple strings that appear in speaker labels. */
function pyRepr(s: string): string {
  if (s.includes("'") && !s.includes('"')) return `"${s}"`;
  return `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

function endsOpen(text: string): boolean {
  let stripped = text.trimEnd();
  while (stripped.length && CLOSERS.includes(stripped[stripped.length - 1]!)) {
    stripped = stripped.slice(0, -1);
  }
  return stripped.length > 0 && !TERMINAL.includes(stripped[stripped.length - 1]!);
}

function startsLower(text: string): boolean {
  for (const ch of text.trimStart()) {
    if (/\p{L}/u.test(ch)) return ch === ch.toLowerCase() && ch !== ch.toUpperCase();
    if (/[0-9]/.test(ch)) return false;
  }
  return false;
}

const wordCount = (text: string): number => text.split(/\s+/).filter(Boolean).length;

/**
 * Does this transcript capitalise sentence starts at all? Raw ASR output is
 * often one lowercase stream, where a lowercase start means nothing.
 */
function usesCapitals(turns: Turn[]): boolean {
  const letters: string[] = [];
  for (const t of turns) {
    const c = t.text.trimStart()[0];
    if (c && /\p{L}/u.test(c)) letters.push(c);
  }
  if (letters.length < 2) return false;
  const uppers = letters.filter((c) => c === c.toUpperCase() && c !== c.toLowerCase()).length;
  return uppers / letters.length >= 0.5;
}

function formatShare(x: number): string {
  // Python's f"{x:.1%}"
  return `${(x * 100).toFixed(1)}%`;
}

/** Flag turn boundaries that look like diarizer mistakes. */
export function diarizationWarnings(turns: TurnsLike, opts: DiagnosticsOptions = {}): Finding[] {
  const {
    maxGapMs = 300,
    flapMaxWords = 4,
    flapMinRun = 4,
    ghostMaxTurns = 2,
    ghostMaxShare = 0.005,
  } = opts;

  const items = [...turnsFrom(turns)];
  const findings: Finding[] = [];
  if (items.length < 2) return findings;

  const capitals = usesCapitals(items);

  // --- mid-utterance flips ---------------------------------------------------
  for (let i = 0; i + 1 < items.length; i++) {
    const prev = items[i]!;
    const cur = items[i + 1]!;
    if (prev.speaker === null || cur.speaker === null || prev.speaker === cur.speaker) continue;
    if (prev.endMs === null || cur.startMs === null) continue;
    const gap = cur.startMs - prev.endMs;
    if (gap > maxGapMs) continue;
    if (!endsOpen(prev.text)) continue;
    const continues = capitals && startsLower(cur.text);
    findings.push({
      kind: "mid_utterance_flip",
      startIndex: prev.index,
      endIndex: cur.index,
      speakers: [prev.speaker, cur.speaker],
      confidence: continues ? "high" : "medium",
      startMs: prev.startMs,
      endMs: cur.endMs,
      reason:
        `${pyRepr(prev.speaker)} stops mid-sentence and ${pyRepr(cur.speaker)} ` +
        `starts ${Math.max(gap, 0)}ms later` +
        (continues ? " with a lowercase continuation" : "") +
        " -- likely one utterance split across two labels",
    });
  }

  // --- flapping --------------------------------------------------------------
  let i = 0;
  const n = items.length;
  while (i < n) {
    const a = items[i]!.speaker;
    if (a === null || wordCount(items[i]!.text) > flapMaxWords) {
      i++;
      continue;
    }
    let j = i + 1;
    let b: string | null = null;
    while (j < n) {
      const t = items[j]!;
      if (t.speaker === null || wordCount(t.text) > flapMaxWords) break;
      if ((j - i) % 2 === 1) {
        if (b === null) {
          if (t.speaker === a) break;
          b = t.speaker;
        } else if (t.speaker !== b) {
          break;
        }
      } else if (t.speaker !== a) {
        break;
      }
      j++;
    }
    const run = j - i;
    if (b !== null && run >= flapMinRun) {
      findings.push({
        kind: "flapping",
        startIndex: items[i]!.index,
        endIndex: items[j - 1]!.index,
        speakers: [a, b],
        confidence: "medium",
        startMs: items[i]!.startMs,
        endMs: items[j - 1]!.endMs,
        reason:
          `${run} consecutive turns of <= ${flapMaxWords} words alternating between ` +
          `${pyRepr(a)} and ${pyRepr(b)} -- diarizers flap like this on a single voice; ` +
          "real back-channel has one long side",
      });
      i = j;
    } else {
      i++;
    }
  }

  // --- ghost speakers --------------------------------------------------------
  const totalChars = items.reduce((acc, t) => acc + t.text.length, 0) || 1;
  const perSpeaker = new Map<string, Turn[]>();
  for (const t of items) {
    if (t.speaker === null) continue;
    const list = perSpeaker.get(t.speaker);
    if (list) list.push(t);
    else perSpeaker.set(t.speaker, [t]);
  }
  if (perSpeaker.size >= 2) {
    for (const [name, ts] of perSpeaker) {
      if (!isGenericLabel(name)) continue;
      const chars = ts.reduce((acc, t) => acc + t.text.length, 0);
      if (ts.length <= ghostMaxTurns && chars / totalChars <= ghostMaxShare) {
        const words = ts.reduce((acc, t) => acc + wordCount(t.text), 0);
        findings.push({
          kind: "ghost_speaker",
          startIndex: ts[0]!.index,
          endIndex: ts[ts.length - 1]!.index,
          speakers: [name],
          confidence: "low",
          startMs: ts[0]!.startMs,
          endMs: ts[ts.length - 1]!.endMs,
          reason:
            `${pyRepr(name)} has ${ts.length} turn(s) and ${words} words, ` +
            `${formatShare(chars / totalChars)} of the transcript -- ` +
            "diarizer-generated labels this sparse are often noise",
        });
      }
    }
  }

  findings.sort((x, y) => x.startIndex - y.startIndex || x.kind.localeCompare(y.kind));
  return findings;
}
