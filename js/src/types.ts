/** Core data types. Mirrors `turnchunk/types.py` exactly. */

import { sha256Hex } from "./sha256.js";

/** One uninterrupted stretch of speech by one speaker. */
export interface Turn {
  text: string;
  /** Resolved identity. `null` means the format did not say. */
  speaker: string | null;
  /**
   * Milliseconds from the start of the recording, or `null`.
   *
   * `null` means "this format did not tell us" and must never be silently
   * turned into `0` -- callers need to distinguish "at the start" from
   * "unknown".
   */
  startMs: number | null;
  endMs: number | null;
  index: number;
  /** The label exactly as it appeared in the source file. */
  rawSpeaker: string | null;
  meta: Record<string, unknown>;
}

export function makeTurn(partial: Partial<Turn> & { text: string }): Turn {
  return {
    speaker: null,
    startMs: null,
    endMs: null,
    index: 0,
    rawSpeaker: null,
    meta: {},
    ...partial,
  };
}

/** A group of consecutive turns, sized for embedding or prompting. */
export class Chunk {
  readonly text: string;
  readonly turns: Turn[];
  /** Indices of turns carried over from the previous chunk as overlap. */
  readonly overlapIndices: number[];
  readonly source: string | null;
  readonly meta: Record<string, unknown>;

  constructor(
    text: string,
    turns: Turn[] = [],
    overlapIndices: number[] = [],
    source: string | null = null,
    meta: Record<string, unknown> = {},
  ) {
    this.text = text;
    this.turns = turns;
    this.overlapIndices = overlapIndices;
    this.source = source;
    this.meta = meta;
  }

  /**
   * Stable content-addressed id, identical to the Python implementation's.
   *
   * Derived from source, turn range and text, so the same input always yields
   * the same id across runs, machines and languages -- which is what makes it
   * usable as a vector-store primary key.
   */
  get id(): string {
    const parts = [this.source ?? "", `${this.turnStart}:${this.turnEnd}`, this.text];
    // NUL-separated, matching Python byte for byte. Any other separator
    // silently yields different ids in the two languages.
    return sha256Hex(parts.join('\x00')).slice(0, 16);
  }

  /** Distinct speakers, in order of first appearance. */
  get speakers(): string[] {
    const seen: string[] = [];
    for (const t of this.turns) {
      if (t.speaker !== null && !seen.includes(t.speaker)) seen.push(t.speaker);
    }
    return seen;
  }

  /** The speaker contributing the most characters, ignoring overlap. */
  get primarySpeaker(): string | null {
    const totals = new Map<string, number>();
    this.turns.forEach((t, i) => {
      if (this.overlapIndices.includes(i) || t.speaker === null) return;
      totals.set(t.speaker, (totals.get(t.speaker) ?? 0) + t.text.length);
    });
    if (totals.size === 0) return this.turns[0]?.speaker ?? null;
    let best: string | null = null;
    let bestN = -1;
    // Ties break on the name, matching Python's max() over (count, name).
    for (const [name, n] of totals) {
      if (n > bestN || (n === bestN && best !== null && name > best)) {
        best = name;
        bestN = n;
      }
    }
    return best;
  }

  get turnStart(): number {
    return this.turns.length ? this.turns[0]!.index : -1;
  }

  get turnEnd(): number {
    return this.turns.length ? this.turns[this.turns.length - 1]!.index : -1;
  }

  get startMs(): number | null {
    for (const t of this.turns) if (t.startMs !== null) return t.startMs;
    return null;
  }

  get endMs(): number | null {
    for (let i = this.turns.length - 1; i >= 0; i--) {
      const v = this.turns[i]!.endMs;
      if (v !== null) return v;
    }
    return null;
  }

  get length(): number {
    return this.text.length;
  }

  toJSON() {
    return {
      id: this.id,
      text: this.text,
      speakers: this.speakers,
      primary_speaker: this.primarySpeaker,
      start_ms: this.startMs,
      end_ms: this.endMs,
      turn_start: this.turnStart,
      turn_end: this.turnEnd,
      overlap_indices: [...this.overlapIndices],
      source: this.source,
      meta: this.meta,
    };
  }
}

/** Parsed transcript: turns plus whatever the parser learned along the way. */
export class Transcript {
  constructor(
    public turns: Turn[] = [],
    public source: string | null = null,
    public format: string | null = null,
    public speakerMap: Record<string, string> = {},
    public meta: Record<string, unknown> = {},
  ) {}

  get speakers(): string[] {
    const seen: string[] = [];
    for (const t of this.turns) {
      if (t.speaker !== null && !seen.includes(t.speaker)) seen.push(t.speaker);
    }
    return seen;
  }

  get hasTimestamps(): boolean {
    return this.turns.some((t) => t.startMs !== null);
  }

  get durationMs(): number | null {
    const ends = this.turns.map((t) => t.endMs).filter((v): v is number => v !== null);
    return ends.length ? Math.max(...ends) : null;
  }

  get length(): number {
    return this.turns.length;
  }

  get text(): string {
    return this.turns.map((t) => t.text).join("\n");
  }

  [Symbol.iterator](): Iterator<Turn> {
    return this.turns[Symbol.iterator]();
  }
}

export type TurnsLike = Transcript | Turn[];

export function turnsFrom(obj: TurnsLike): Turn[] {
  return obj instanceof Transcript ? obj.turns : obj;
}
