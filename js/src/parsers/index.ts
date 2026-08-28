/**
 * Format detection and dispatch. Mirrors `turnchunk/parsers/__init__.py`.
 *
 * `parse()` never asks the caller what format a file is. Detection runs on
 * content, not on the file extension, because exports are routinely saved with
 * the wrong suffix.
 */

import type { Transcript } from "../types.js";
import { looksLikeAsrJson, parseAsrJson } from "./asr.js";
import { looksLikePlain, parsePlain } from "./plain.js";
import { looksLikeSrt, parseSrt } from "./srt.js";
import { looksLikeVtt, parseVtt } from "./vtt.js";
import type { ParseOptions } from "./vtt.js";

export class UnknownFormatError extends Error {
  override readonly name = "UnknownFormatError";
}

type Detector = (text: string) => boolean;
type Parser = (text: string, opts: ParseOptions) => Transcript;

// Ordered most-specific first. JSON has an unambiguous signature so it can
// never be misread as plain text, whereas plain-text detection is fuzzy.
const DETECTORS: Array<[string, Detector, Parser]> = [
  ["json", looksLikeAsrJson, parseAsrJson],
  ["vtt", looksLikeVtt, parseVtt],
  ["srt", looksLikeSrt, parseSrt],
  ["plain", looksLikePlain, parsePlain],
];

export const FORMATS = DETECTORS.map(([name]) => name);

/** Return the detected format name, or null. */
export function detectFormat(text: string): string | null {
  for (const [name, detector] of DETECTORS) {
    if (detector(text)) return name;
  }
  return null;
}

export interface TopLevelParseOptions extends ParseOptions {
  format?: string | null;
}

/**
 * Parse a transcript from text.
 *
 * Unlike the Python package this takes a string rather than a path, because
 * the same build has to run in a browser and on an edge runtime where there is
 * no filesystem. Read the file yourself and pass the contents.
 */
export function parse(text: string, opts: TopLevelParseOptions = {}): Transcript {
  const { format = null, ...rest } = opts;
  const chosen = format ?? detectFormat(text);
  if (chosen === null) {
    throw new UnknownFormatError(
      `Could not detect the transcript format. Pass format explicitly (one of: ${FORMATS.join(", ")}).`,
    );
  }
  for (const [name, , parser] of DETECTORS) {
    if (name === chosen) return parser(text, rest);
  }
  throw new UnknownFormatError(`Unknown format '${chosen}'. Known: ${FORMATS.join(", ")}`);
}

export { parseAsrJson, parsePlain, parseSrt, parseVtt };
export type { ParseOptions };
