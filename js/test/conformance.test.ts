/**
 * Cross-language conformance.
 *
 * `tests/corpus/conformance.json` is generated from the Python implementation
 * and checked into the repo. This suite asserts the TypeScript port produces
 * byte-identical results for every case in it -- same turns, same chunk text,
 * same chunk ids, same speaker resolution, same sentence boundaries.
 *
 * Regenerate the corpus with `python scripts/build_corpus.py`.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { buildSpeakerMap, chunk, parse, splitSentences } from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");

interface Corpus {
  version: number;
  parse: Record<string, { format: string; turns: ExpectedTurn[] }>;
  chunk: Record<string, Array<{ options: ChunkOpts; chunks: ExpectedChunk[] }>>;
  sentences: Array<{ input: string; output: string[] }>;
  speakers: Array<{ input: string[]; output: Record<string, string> }>;
}
interface ExpectedTurn {
  text: string;
  speaker: string | null;
  start_ms: number | null;
  end_ms: number | null;
  index: number;
}
interface ChunkOpts {
  target: number;
  overlap: number;
  min_tail_ratio: number;
}
interface ExpectedChunk {
  id: string;
  text: string;
  speakers: string[];
  primary_speaker: string | null;
  start_ms: number | null;
  end_ms: number | null;
  turn_start: number;
  turn_end: number;
  overlap_indices: number[];
}

const corpus: Corpus = JSON.parse(
  readFileSync(join(ROOT, "tests", "corpus", "conformance.json"), "utf8"),
);

const fixture = (name: string) =>
  readFileSync(join(ROOT, "tests", "fixtures", name), "utf8");

describe("corpus", () => {
  it("is present and non-trivial", () => {
    expect(corpus.version).toBe(1);
    expect(Object.keys(corpus.parse).length).toBeGreaterThan(10);
  });
});

describe("parse matches Python", () => {
  for (const [name, expected] of Object.entries(corpus.parse)) {
    it(name, () => {
      const t = parse(fixture(name), { source: name });
      expect(t.format).toBe(expected.format);
      expect(t.turns.length).toBe(expected.turns.length);
      t.turns.forEach((turn, i) => {
        const e = expected.turns[i]!;
        expect({
          text: turn.text,
          speaker: turn.speaker,
          start_ms: turn.startMs,
          end_ms: turn.endMs,
          index: turn.index,
        }).toEqual(e);
      });
    });
  }
});

describe("chunk matches Python", () => {
  for (const [name, cases] of Object.entries(corpus.chunk)) {
    for (const [n, c] of cases.entries()) {
      it(`${name} [case ${n}] target=${c.options.target} overlap=${c.options.overlap}`, () => {
        const t = parse(fixture(name), { source: name });
        const got = chunk(t, {
          target: c.options.target,
          overlap: c.options.overlap,
          minTailRatio: c.options.min_tail_ratio,
          source: name,
        });
        expect(got.length).toBe(c.chunks.length);
        got.forEach((chk, i) => {
          const e = c.chunks[i]!;
          expect({
            id: chk.id,
            text: chk.text,
            speakers: chk.speakers,
            primary_speaker: chk.primarySpeaker,
            start_ms: chk.startMs,
            end_ms: chk.endMs,
            turn_start: chk.turnStart,
            turn_end: chk.turnEnd,
            overlap_indices: chk.overlapIndices,
          }).toEqual(e);
        });
      });
    }
  }
});

describe("sentence splitting matches Python", () => {
  corpus.sentences.forEach((c, i) => {
    it(`case ${i}: ${JSON.stringify(c.input.slice(0, 40))}`, () => {
      const out = splitSentences(c.input);
      expect(out).toEqual(c.output);
      expect(out.join("")).toBe(c.input); // never loses a character
    });
  });
});

describe("speaker resolution matches Python", () => {
  corpus.speakers.forEach((c, i) => {
    it(`case ${i}: ${c.input.join(", ")}`, () => {
      expect(buildSpeakerMap(c.input)).toEqual(c.output);
    });
  });
});

describe("chunk ids are identical across languages", () => {
  it("every id in the corpus is reproduced exactly", () => {
    let checked = 0;
    for (const [name, cases] of Object.entries(corpus.chunk)) {
      for (const c of cases) {
        const t = parse(fixture(name), { source: name });
        const got = chunk(t, {
          target: c.options.target,
          overlap: c.options.overlap,
          minTailRatio: c.options.min_tail_ratio,
          source: name,
        });
        got.forEach((chk, i) => {
          expect(chk.id).toBe(c.chunks[i]!.id);
          checked++;
        });
      }
    }
    expect(checked).toBeGreaterThan(30);
  });
});
