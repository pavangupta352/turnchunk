/**
 * The chunker. Mirrors `turnchunk/chunker.py`.
 *
 * Four rules, in the order they are applied:
 *
 * 1. Oversized turns split internally at sentence boundaries, every piece
 *    keeping its speaker.
 * 2. A chunk boundary may only fall at a turn boundary.
 * 3. Overlap respects turns, and carried-over turns are marked.
 * 4. Short tails merge backwards rather than becoming stubs.
 *
 * Sizes are measured on the *rendered* text -- the string that will actually be
 * embedded, speaker label included -- because that is what has to fit.
 */

import { splitSentences } from "./sentences.js";
import { Chunk, Transcript, makeTurn, turnsFrom } from "./types.js";
import type { Turn, TurnsLike } from "./types.js";

export type SizeFn = (text: string) => number;

export const DEFAULT_TEMPLATE = "{speaker}: {text}";
export const DEFAULT_SEPARATOR = "\n";
export const UNKNOWN_SPEAKER = "UNKNOWN";

export interface ChunkOptions {
  target?: number;
  overlap?: number;
  minTailRatio?: number;
  sizeFn?: SizeFn;
  template?: string;
  separator?: string;
  source?: string | null;
  maxOverflow?: number;
}

/** Render one turn to the text that will be embedded. */
export function renderTurn(turn: Turn, template: string = DEFAULT_TEMPLATE): string {
  return template
    .replace("{speaker}", turn.speaker ?? UNKNOWN_SPEAKER)
    .replace("{text}", turn.text)
    .replace("{start_ms}", turn.startMs === null ? "" : String(turn.startMs))
    .replace("{end_ms}", turn.endMs === null ? "" : String(turn.endMs));
}

function interpolate(turn: Turn, lo: number, hi: number): [number | null, number | null] {
  if (turn.startMs === null || turn.endMs === null) return [turn.startMs, turn.endMs];
  const span = turn.endMs - turn.startMs;
  return [Math.trunc(turn.startMs + span * lo), Math.trunc(turn.startMs + span * hi)];
}

function hardSplit(text: string, target: number, sizeFn: SizeFn): string[] {
  const out: string[] = [];
  let buf = "";
  for (const word of text.split(" ")) {
    const candidate = buf ? `${buf} ${word}` : word;
    if (buf && sizeFn(candidate) > target) {
      out.push(buf + " ");
      buf = word;
    } else {
      buf = candidate;
    }
  }
  if (buf) out.push(buf);
  return out.length ? out : [text];
}

function cutAt(text: string, budget: number, sizeFn: SizeFn): [string, string] {
  let lo = 1;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    if (sizeFn(text.slice(0, mid)) <= budget) lo = mid;
    else hi = mid - 1;
  }
  let cut = lo;
  const space = text.lastIndexOf(" ", cut);
  if (space > cut * 0.6) cut = space + 1;
  return [text.slice(0, cut), text.slice(cut)];
}

/** Rule 1: break a too-long turn at sentence boundaries, keeping the speaker. */
export function splitOversizedTurn(
  turn: Turn,
  target: number,
  sizeFn: SizeFn,
  template: string,
): Turn[] {
  if (sizeFn(renderTurn(turn, template)) <= target) return [turn];

  let sentences = splitSentences(turn.text);
  if (sentences.length <= 1) sentences = hardSplit(turn.text, target, sizeFn);

  const overhead = sizeFn(
    renderTurn(makeTurn({ text: "", speaker: turn.speaker }), template),
  );
  const budget = Math.max(1, target - overhead);

  const pieces: string[] = [];
  let buf = "";
  for (const s of sentences) {
    if (buf && sizeFn(buf + s) > budget) {
      pieces.push(buf);
      buf = s;
    } else {
      buf += s;
    }
    while (sizeFn(buf) > budget) {
      const [head, rest] = cutAt(buf, budget, sizeFn);
      pieces.push(head);
      buf = rest;
    }
  }
  if (buf) pieces.push(buf);

  const total = pieces.reduce((a, p) => a + p.length, 0) || 1;
  const out: Turn[] = [];
  let cursor = 0;
  pieces.forEach((p, i) => {
    const lo = cursor / total;
    cursor += p.length;
    const hi = cursor / total;
    const [startMs, endMs] = interpolate(turn, lo, hi);
    const meta: Record<string, unknown> = { ...turn.meta, part: [i, pieces.length] };
    // The first piece keeps the turn's real start and the last its real end;
    // every boundary between them is interpolated, so mark those as estimates.
    const interior = i !== 0 || i !== pieces.length - 1;
    if (turn.startMs !== null && pieces.length > 1 && interior) meta["time_estimated"] = true;
    out.push(
      makeTurn({
        text: p.trim() || p,
        speaker: turn.speaker,
        startMs,
        endMs,
        index: turn.index,
        rawSpeaker: turn.rawSpeaker,
        meta,
      }),
    );
  });
  return out;
}

function measure(indices: number[], sizes: number[], sepSize: number): number {
  if (!indices.length) return 0;
  return indices.reduce((a, i) => a + sizes[i]!, 0) + sepSize * (indices.length - 1);
}

/**
 * Choose whole turns from the end of `emitted` to repeat as overlap.
 *
 * Never carries the entire chunk -- that would make no forward progress and
 * loop forever -- and never more than half the target, which would leave no
 * room for new content.
 */
function carry(
  emitted: number[],
  sizes: number[],
  sepSize: number,
  overlap: number,
  target: number,
): number[] {
  if (overlap <= 0 || emitted.length <= 1) return [];
  const cap = Math.min(overlap, Math.floor(target / 2));
  const carried: number[] = [];
  let total = 0;
  for (let k = emitted.length - 1; k >= 1; k--) {
    const i = emitted[k]!;
    const addition = sizes[i]! + (carried.length ? sepSize : 0);
    if (total + addition > cap && carried.length) break;
    carried.unshift(i);
    total += addition;
    if (total >= cap) break;
  }
  return carried;
}

function overlapCount(group: number[], groups: number[][], gi: number): number {
  if (gi === 0) return 0;
  const prev = new Set(groups[gi - 1]!);
  let n = 0;
  for (const i of group) {
    if (prev.has(i)) n++;
    else break;
  }
  return n;
}

function build(
  group: number[],
  units: Turn[],
  rendered: string[],
  separator: string,
  source: string | null,
  overlapN: number,
): Chunk {
  return new Chunk(
    group.map((i) => rendered[i]!).join(separator),
    group.map((i) => units[i]!),
    Array.from({ length: overlapN }, (_, k) => k),
    source,
  );
}

/** Chunk a transcript at speaker-turn boundaries. */
export function chunk(turns: TurnsLike, options: ChunkOptions = {}): Chunk[] {
  const {
    target = 2000,
    overlap = 0,
    minTailRatio = 1 / 3,
    sizeFn = (s: string) => s.length,
    template = DEFAULT_TEMPLATE,
    separator = DEFAULT_SEPARATOR,
    maxOverflow = 1.5,
  } = options;

  if (target <= 0) throw new RangeError("target must be positive");
  if (overlap < 0) throw new RangeError("overlap must not be negative");
  if (overlap >= target) throw new RangeError("overlap must be smaller than target");
  if (!(minTailRatio >= 0 && minTailRatio < 1)) {
    throw new RangeError("minTailRatio must be in [0, 1)");
  }

  const items = turnsFrom(turns);
  let source = options.source ?? null;
  if (options.source === undefined && turns instanceof Transcript) source = turns.source;
  if (!items.length) return [];

  // Rule 1 -- explode oversized turns first, so packing only sees units that
  // can fit on their own.
  const units: Turn[] = [];
  for (const t of items) units.push(...splitOversizedTurn(t, target, sizeFn, template));

  const rendered = units.map((u) => renderTurn(u, template));
  const sizes = rendered.map(sizeFn);
  const sepSize = sizeFn(separator);

  // Rules 2 and 3 -- greedy packing on turn boundaries, turn-aligned overlap.
  const groups: number[][] = [];
  let current: number[] = [];
  let currentSize = 0;

  for (let i = 0; i < units.length; i++) {
    let addition = sizes[i]! + (current.length ? sepSize : 0);
    if (current.length && currentSize + addition > target) {
      groups.push(current);
      current = [...carry(current, sizes, sepSize, overlap, target)];
      currentSize = measure(current, sizes, sepSize);
      addition = sizes[i]! + (current.length ? sepSize : 0);
    }
    current.push(i);
    currentSize += addition;
  }
  if (current.length) groups.push(current);

  let chunks = groups.map((g, gi) =>
    build(g, units, rendered, separator, source, overlapCount(g, groups, gi)),
  );

  // Rule 4 -- merge a stub tail backwards.
  if (chunks.length >= 2 && minTailRatio > 0) {
    const lastGroup = groups[groups.length - 1]!;
    const overlapN = chunks[chunks.length - 1]!.overlapIndices.length;
    const own = lastGroup.slice(overlapN);
    if (!own.length) {
      // The tail is nothing but overlap -- it carries no new content at all.
      return chunks.slice(0, -1);
    }
    const ownSize = sizeFn(own.map((i) => rendered[i]!).join(separator));
    if (ownSize < target * minTailRatio) {
      const merged = [...groups[groups.length - 2]!, ...own];
      const mergedText = merged.map((i) => rendered[i]!).join(separator);
      if (sizeFn(mergedText) <= target * maxOverflow) {
        const keptOverlap = chunks[chunks.length - 2]!.overlapIndices.length;
        chunks[chunks.length - 2] = build(
          merged, units, rendered, separator, source, keptOverlap,
        );
        chunks = chunks.slice(0, -1);
      }
    }
  }

  return chunks;
}
