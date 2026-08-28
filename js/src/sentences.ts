/**
 * Dependency-free sentence segmentation. Mirrors `turnchunk/sentences.py`.
 *
 * Used only to split a turn too long to fit, so it fails safe: when unsure it
 * does not split, because an over-long chunk is a much smaller problem than a
 * chunk that begins mid-clause.
 */

const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
  "gen", "col", "lt", "sgt", "capt", "cmdr", "adm", "gov", "pres", "supt",
  "rep", "sen", "atty", "asst", "assoc", "dept", "univ", "inc", "ltd", "co",
  "corp", "llc", "plc", "vs", "etc", "eg", "ie", "al", "approx", "est",
  "fig", "vol", "no", "pp", "ed", "cf", "viz", "ca", "circa",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
  "sat", "sun",
  "am", "pm", "ph", "phd", "md", "ba", "ma", "bsc", "msc", "llb",
  "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m",
]);

const CJK_TERMINATORS = "。！？";

// term / closing quotes / trailing whitespace. Whitespace is required for
// Latin terminators only -- CJK does not use it, and demanding it would mean
// Japanese and Chinese transcripts are never segmented at all.
const BOUNDARY =
  /(?<term>[.!?…。！？]+)(?<close>["'’”)\]}」』]*)(?<space>\s*)/gu;

const LOOKBACK = 40;
const INITIAL = /(?:^|\s)[A-Z]\.$/u;
const ABBREV_TAIL = /([A-Za-z][A-Za-z.]*)\.$/u;

function isAbbreviation(before: string): boolean {
  if (INITIAL.test(before)) return true;
  const m = ABBREV_TAIL.exec(before);
  if (!m) return false;
  const word = m[1]!.toLowerCase().replace(/\.+$/, "");
  if (ABBREVIATIONS.has(word)) return true;
  // A single letter before a period, as inside "U.S." -- treat the whole
  // dotted run as an abbreviation rather than a sentence end.
  return word.length === 1;
}

/** Does this text look like ASR output with no capitalisation? */
function isUncapitalised(text: string): boolean {
  const sample = text.slice(0, 20_000);
  let letters = 0;
  let uppers = 0;
  for (const c of sample) {
    if (!/\p{L}/u.test(c)) continue;
    letters++;
    if (c !== c.toLowerCase() && c === c.toUpperCase()) uppers++;
  }
  if (letters < 40) return false; // too short to judge; stay conservative
  return uppers / letters < 0.01;
}

function looksLikeStart(ch: string): boolean {
  if (!ch) return false;
  if (/\p{Lu}/u.test(ch) || /[0-9]/.test(ch)) return true;
  if ("\"'“‘([{".includes(ch)) return true;
  // Non-cased scripts (CJK, Devanagari, Arabic...) have no capitals, so
  // requiring one would mean never splitting them.
  return /\p{L}/u.test(ch) && ch.toLowerCase() === ch.toUpperCase();
}

/**
 * Spans covering `text` exactly: concatenating every slice reproduces the
 * input character for character.
 */
export function sentenceSpans(
  text: string,
  opts: { allowLowercaseStart?: boolean } = {},
): Array<[number, number]> {
  if (!text) return [];
  const allowLowercase = opts.allowLowercaseStart ?? isUncapitalised(text);

  const spans: Array<[number, number]> = [];
  let start = 0;

  // Everything in this loop must be O(1) in document length. Slicing the
  // remainder of the document here is quadratic, and was a real bug: 1.4s to
  // chunk a 260 KB transcript.
  BOUNDARY.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = BOUNDARY.exec(text)) !== null) {
    if (m[0].length === 0) {
      BOUNDARY.lastIndex++;
      continue;
    }
    const g = m.groups!;
    const term = g["term"]!;
    const close = g["close"]!;
    const space = g["space"]!;

    const termStart = m.index;
    const cut = termStart + term.length + close.length;
    const end = cut + space.length;
    const isCjk = CJK_TERMINATORS.includes(term[0]!);
    const isPeriod = term === ".";

    // Latin terminators need whitespace after them, or "example.com" and
    // "3.5x" would both become sentence breaks.
    if (!isCjk && !space) continue;

    const nextChar = end < text.length ? text[end]! : "";

    // "3.14" / "v1.2" -- a period between digits is never a boundary.
    if (
      isPeriod &&
      /[0-9]/.test(nextChar) &&
      termStart > 0 &&
      /[0-9]/.test(text[termStart - 1]!)
    ) {
      continue;
    }

    // "Dr. Smith", "e.g. this", "J. Smith"
    if (isPeriod && isAbbreviation(text.slice(Math.max(start, cut - LOOKBACK), cut))) {
      continue;
    }

    // A bare period should be followed by something that looks like a start.
    if (isPeriod && nextChar && !allowLowercase && !looksLikeStart(nextChar)) continue;

    if (end > start) {
      spans.push([start, end]);
      start = end;
    }
  }

  if (start < text.length) spans.push([start, text.length]);
  return spans;
}

/** Split `text` into sentences, preserving all whitespace. */
export function splitSentences(
  text: string,
  opts: { allowLowercaseStart?: boolean } = {},
): string[] {
  return sentenceSpans(text, opts).map(([s, e]) => text.slice(s, e));
}
