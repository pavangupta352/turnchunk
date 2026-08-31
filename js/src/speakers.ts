/**
 * Speaker identity resolution. Mirrors `turnchunk/speakers.py`.
 *
 * The two failure modes are not symmetric: leaving one person as two labels is
 * untidy, but merging two different people silently misattributes what they
 * said. So a merge only happens when it is unambiguous, and the full mapping is
 * always returned for inspection.
 */

import { turnsFrom } from "./types.js";
import type { Turn, TurnsLike } from "./types.js";

// "John (Host)", "John [interviewer]", "John - Acme Corp"
const TRAILING_ROLE = /\s*[([{][\s\S]*?[)\]}]\s*$|\s+[-–—]\s+[\s\S]*$/;
// Generic diarization labels: SPEAKER_00, Speaker 1, spk0, S1
const GENERIC = /^(?:speaker|spk|s)[\s_-]*(\d+)$/i;
const PUNCT = /[^\p{L}\p{N}_\s]/gu;

/** Fold a label to a comparison key: accents, case and punctuation removed. */
export function canonicalKey(name: string): string {
  const stripped = name.normalize("NFKD").replace(/\p{M}/gu, "");
  return stripped.replace(PUNCT, " ").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

/** Is this a diarizer's placeholder rather than a human name? */
export function isGenericLabel(name: string): boolean {
  return GENERIC.test(name.trim());
}

/** Remove a trailing role or affiliation: `"John (Host)"` -> `"John"`. */
export function stripRole(name: string): string {
  const cleaned = name.trim().replace(TRAILING_ROLE, "").trim();
  return cleaned || name.trim();
}

const tokens = (key: string): string[] => key.split(" ").filter(Boolean);

/** Does `short` look like an abbreviation of `full`? */
function isInitialForm(short: string[], full: string[]): boolean {
  if (short.length !== full.length || !short.length) return false;
  return short.every((a, i) => {
    const b = full[i]!;
    if (a === b) return true;
    return a.length === 1 && b.startsWith(a);
  });
}

/** Could `shortKey` be the same person as the longer `fullKey`? */
function compatible(shortKey: string, fullKey: string): boolean {
  const s = tokens(shortKey);
  const f = tokens(fullKey);
  if (!s.length || !f.length || shortKey === fullKey) return false;
  if (s.length > f.length) return false;
  if (isInitialForm(s, f)) return true;
  // "john" is contained in "john smith" -- a strict subset of its tokens.
  return s.length < f.length && s.every((t) => f.includes(t));
}

/**
 * Choose the label a human would recognise: a properly-capitalised form over
 * ALL CAPS or all lowercase, then the most frequent, then the longest.
 */
function pickDisplay(members: string[], counts: Record<string, number>): string {
  let best = members[0]!;
  let bestScore: [number, number, number] = [-1, -1, -1];
  for (const name of members) {
    const cleaned = stripRole(name);
    const titlecase =
      cleaned !== cleaned.toUpperCase() && cleaned !== cleaned.toLowerCase() ? 1 : 0;
    const score: [number, number, number] = [titlecase, counts[name] ?? 0, cleaned.length];
    if (
      score[0] > bestScore[0] ||
      (score[0] === bestScore[0] && score[1] > bestScore[1]) ||
      (score[0] === bestScore[0] && score[1] === bestScore[1] && score[2] > bestScore[2])
    ) {
      best = name;
      bestScore = score;
    }
  }
  return stripRole(best);
}

/**
 * Above this many distinct labels, partial-name merging is skipped.
 *
 * The merge is inherently a comparison between labels, and a transcript with
 * thousands of distinct speakers is a misparse or a hostile file rather than a
 * real conversation. Case and punctuation folding still runs.
 */
export const MAX_MERGE_LABELS = 1000;

export interface SpeakerMapOptions {
  counts?: Record<string, number>;
  mergePartialNames?: boolean;
  maxMergeLabels?: number;
}

/** Map every raw label to its resolved display name. */
export function buildSpeakerMap(
  names: Iterable<string>,
  opts: SpeakerMapOptions = {},
): Record<string, string> {
  const {
    counts = {},
    mergePartialNames = true,
    maxMergeLabels = MAX_MERGE_LABELS,
  } = opts;
  const raw = [...new Set([...names].filter(Boolean))];
  if (!raw.length) return {};

  // Pass 1 -- fold case, accents, punctuation and trailing roles.
  const groups = new Map<string, string[]>();
  for (const name of raw) {
    const key = canonicalKey(stripRole(name));
    const list = groups.get(key);
    if (list) list.push(name);
    else groups.set(key, [name]);
  }

  const display = new Map<string, string>();
  for (const [key, members] of groups) display.set(key, pickDisplay(members, counts));

  // Pass 2 -- merge partial names into their unambiguous full form. Generic
  // diarizer labels are never merged: SPEAKER_00 and SPEAKER_01 are different
  // people by definition.
  if (mergePartialNames && groups.size <= maxMergeLabels) {
    const human = [...groups.keys()].filter((k) => !isGenericLabel(display.get(k)!));

    // Compare each label only against labels sharing a token with it.
    // Comparing every label against every other is quadratic, and a crafted
    // upload manufacturing thousands of "speakers" turned that into seconds
    // of CPU from a single file.
    const byToken = new Map<string, string[]>();
    for (const key of human) {
      for (const token of new Set(tokens(key))) {
        const list = byToken.get(token);
        if (list) list.push(key);
        else byToken.set(token, [key]);
      }
    }

    const alias = new Map<string, string>();
    for (const short of human) {
      const seen = new Set<string>();
      for (const token of new Set(tokens(short))) {
        for (const k of byToken.get(token) ?? []) seen.add(k);
      }
      // Initial forms ("j smith") share the surname token with the full name,
      // so the shared-token index still reaches them.
      const candidates = [...seen].filter((f) => f !== short && compatible(short, f));
      // Only merge when there is exactly one possible match. "J. Smith" with
      // both "John Smith" and "Jane Smith" present stays separate.
      if (candidates.length === 1) alias.set(short, candidates[0]!);
    }
    // Resolve chains without cycling.
    for (const short of [...alias.keys()]) {
      const seen = new Set([short]);
      let target = alias.get(short)!;
      while (alias.has(target) && !seen.has(target)) {
        seen.add(target);
        target = alias.get(target)!;
      }
      alias.set(short, target);
    }
    for (const [short, target] of alias) display.set(short, display.get(target)!);
  }

  const out: Record<string, string> = {};
  for (const name of raw) out[name] = display.get(canonicalKey(stripRole(name)))!;
  return out;
}

export interface ResolveOptions {
  mergePartialNames?: boolean;
  maxMergeLabels?: number;
  rename?: Record<string, string>;
}

/** Unify speaker labels across a transcript. */
export function resolveSpeakers(
  turns: TurnsLike,
  opts: ResolveOptions = {},
): { turns: Turn[]; mapping: Record<string, string> } {
  const { mergePartialNames = true, maxMergeLabels = MAX_MERGE_LABELS, rename } = opts;
  const items = turnsFrom(turns);

  const counts: Record<string, number> = {};
  for (const t of items) {
    const name = t.rawSpeaker ?? t.speaker;
    if (name) counts[name] = (counts[name] ?? 0) + 1;
  }

  const mapping = buildSpeakerMap(Object.keys(counts), {
    counts,
    mergePartialNames,
    maxMergeLabels,
  });

  if (rename) {
    // Overrides apply to both the raw label and its resolved form.
    for (const [raw, resolved] of Object.entries({ ...mapping })) {
      if (raw in rename) mapping[raw] = rename[raw]!;
      else if (resolved in rename) mapping[raw] = rename[resolved]!;
    }
    for (const [raw, target] of Object.entries(rename)) {
      if (!(raw in mapping)) mapping[raw] = target;
    }
  }

  for (const t of items) {
    const original = t.rawSpeaker ?? t.speaker;
    if (original === null) continue;
    t.rawSpeaker = original;
    t.speaker = mapping[original] ?? original;
  }
  return { turns: items, mapping };
}
