/**
 * turnchunk -- chunking that understands who is speaking.
 *
 * ```ts
 * import { parse, chunk } from "turnchunk";
 *
 * const turns = parse(vttText);              // format auto-detected
 * const chunks = chunk(turns, { target: 2000, overlap: 200 });
 *
 * for (const c of chunks) {
 *   console.log(c.primarySpeaker, c.startMs, c.text.slice(0, 80));
 * }
 * ```
 *
 * Zero dependencies. Every chunk boundary falls on a speaker turn boundary.
 * Behaviour is verified identical to the Python package against a shared
 * conformance corpus.
 */

export const VERSION = "0.1.0";

export {
  DEFAULT_SEPARATOR,
  DEFAULT_TEMPLATE,
  UNKNOWN_SPEAKER,
  chunk,
  renderTurn,
  splitOversizedTurn,
} from "./chunker.js";
export type { ChunkOptions, SizeFn } from "./chunker.js";

export {
  FORMATS,
  UnknownFormatError,
  detectFormat,
  parse,
  parseAsrJson,
  parsePlain,
  parseSrt,
  parseVtt,
} from "./parsers/index.js";
export type { ParseOptions, TopLevelParseOptions } from "./parsers/index.js";

export { sentenceSpans, splitSentences } from "./sentences.js";

export {
  buildSpeakerMap,
  canonicalKey,
  isGenericLabel,
  resolveSpeakers,
  stripRole,
} from "./speakers.js";
export type { ResolveOptions, SpeakerMapOptions } from "./speakers.js";

export { Chunk, Transcript, makeTurn, turnsFrom } from "./types.js";
export type { Turn, TurnsLike } from "./types.js";

export { formatTimestamp, toContext, toContextAll } from "./render.js";
