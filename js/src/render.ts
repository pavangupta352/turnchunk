/**
 * Rendering chunks for an LLM prompt. Mirrors `turnchunk/render.py`.
 *
 * Everyone writes this by hand and everyone gets a detail subtly wrong --
 * timestamps in raw milliseconds, speakers dropped from continuation lines, or
 * overlap presented as though it were new text.
 */

import type { Chunk } from "./types.js";

/** Milliseconds to `H:MM:SS` / `MM:SS`. Unknown time renders `--:--`. */
export function formatTimestamp(ms: number | null, alwaysHours = false): string {
  if (ms === null) return "--:--";
  const total = Math.floor(Math.max(0, ms) / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h || alwaysHours ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

export interface ContextOptions {
  timestamps?: boolean;
  header?: boolean;
  includeOverlap?: boolean;
}

/** Render a chunk as text to put in a prompt. */
export function toContext(chunk: Chunk, opts: ContextOptions = {}): string {
  const { timestamps = true, header = true, includeOverlap = true } = opts;
  const lines: string[] = [];
  if (header) {
    const span = `${formatTimestamp(chunk.startMs)}-${formatTimestamp(chunk.endMs)}`;
    lines.push(`[${chunk.source ?? "transcript"} ${span}]`);
  }
  chunk.turns.forEach((turn, i) => {
    if (!includeOverlap && chunk.overlapIndices.includes(i)) return;
    const prefix = timestamps ? `(${formatTimestamp(turn.startMs)}) ` : "";
    lines.push(`${prefix}${turn.speaker ?? "UNKNOWN"}: ${turn.text}`);
  });
  return lines.join("\n");
}

/** Render several chunks into one prompt block, dropping repeated overlap. */
export function toContextAll(
  chunks: Iterable<Chunk>,
  opts: ContextOptions & { separator?: string } = {},
): string {
  const { separator = "\n\n---\n\n", ...rest } = opts;
  const options = { includeOverlap: false, ...rest };
  return [...chunks].map((c) => toContext(c, options)).join(separator);
}
