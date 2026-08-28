/**
 * Speech-to-text vendor JSON. Mirrors `turnchunk/parsers/asr.py`.
 *
 * The difference that silently ruins a pipeline is time units: Whisper,
 * Deepgram, Rev and Speechmatics emit seconds as floats, AssemblyAI emits
 * integer milliseconds. Read one as the other and the text is perfect while
 * every citation points at the wrong moment, forever.
 */

import { Transcript, makeTurn } from "../types.js";
import type { Turn } from "../types.js";
import { mergeTurns } from "./base.js";
import type { ParseOptions } from "./vtt.js";

type Json = any;

const VENDORS = [
  "whisper", "assemblyai", "deepgram", "rev", "speechmatics",
  "generic_list", "generic_dict",
] as const;
export type Vendor = (typeof VENDORS)[number];

function isTurnList(items: Json): boolean {
  if (!Array.isArray(items) || !items.length) return false;
  const first = items[0];
  if (typeof first !== "object" || first === null) return false;
  return ["text", "transcript", "value", "content"].some((k) => k in first);
}

export function identify(data: Json): Vendor | null {
  if (Array.isArray(data)) return isTurnList(data) ? "generic_list" : null;
  if (typeof data !== "object" || data === null) return null;
  if ("monologues" in data) return "rev";
  const results = data["results"];
  if (results && !Array.isArray(results) && typeof results === "object" && "channels" in results) {
    return "deepgram";
  }
  if (Array.isArray(results) && results.length && results[0] && "alternatives" in results[0]) {
    return "speechmatics";
  }
  if (Array.isArray(data["utterances"])) {
    return "words" in data || "id" in data ? "assemblyai" : "generic_dict";
  }
  if (Array.isArray(data["segments"])) return "whisper";
  for (const key of ["turns", "transcript", "items", "entries"]) {
    if (isTurnList(data[key])) return "generic_dict";
  }
  return null;
}

export function looksLikeAsrJson(text: string): boolean {
  const stripped = text.replace(/^[﻿\s]+/, "");
  if (!stripped.startsWith("{") && !stripped.startsWith("[")) return false;
  try {
    return identify(JSON.parse(stripped)) !== null;
  } catch {
    return false;
  }
}

const secToMs = (v: Json): number | null => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n * 1000) : null;
};

const asMs = (v: Json): number | null => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n) : null;
};

/** Normalise a speaker field. Integer ids become `SPEAKER_00`-style labels. */
function speakerOf(value: Json, prefix = "SPEAKER"): string | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "boolean") return null;
  if (typeof value === "number" && Number.isInteger(value)) {
    return `${prefix}_${String(value).padStart(2, "0")}`;
  }
  const text = String(value).trim();
  if (!text) return null;
  if (/^\d+$/.test(text)) return `${prefix}_${String(parseInt(text, 10)).padStart(2, "0")}`;
  return text;
}

interface WordOpts {
  textKey: string;
  fallbackKey?: string;
  toMs: (v: Json) => number | null;
  joinPunct?: boolean;
}

/** Collapse word-level output into turns, breaking when the speaker changes. */
function groupWords(words: Json[], o: WordOpts): Turn[] {
  const out: Turn[] = [];
  let current: Turn | null = null;
  for (const w of words) {
    if (typeof w !== "object" || w === null) continue;
    const token = String(w[o.textKey] ?? (o.fallbackKey ? w[o.fallbackKey] : "") ?? "");
    if (!token) continue;
    const spk = speakerOf(w["speaker"]);
    const start = o.toMs(w["start"]);
    const end = o.toMs(w["end"]);
    const isPunct = Boolean(o.joinPunct) && w["type"] === "punctuation";

    if (current === null || (spk !== current.speaker && !isPunct)) {
      current = makeTurn({ text: token, speaker: spk, startMs: start, endMs: end, index: out.length });
      out.push(current);
    } else {
      current.text += (isPunct ? "" : " ") + token;
      if (end !== null) current.endMs = end;
    }
  }
  for (const t of out) t.text = t.text.trim();
  return out.filter((t) => t.text);
}

function fromWhisper(data: Json): Turn[] {
  const out: Turn[] = [];
  for (const seg of data["segments"] ?? []) {
    if (typeof seg !== "object" || seg === null) continue;
    const text = String(seg["text"] ?? "").trim();
    if (!text) continue;
    out.push(makeTurn({
      text, speaker: speakerOf(seg["speaker"]),
      startMs: secToMs(seg["start"]), endMs: secToMs(seg["end"]), index: out.length,
    }));
  }
  return out;
}

/** AssemblyAI. Times are integer milliseconds, not seconds. */
function fromAssemblyAI(data: Json): Turn[] {
  const utterances = data["utterances"] ?? [];
  if (utterances.length) {
    const out: Turn[] = [];
    for (const u of utterances) {
      const text = String(u["text"] ?? "").trim();
      if (!text) continue;
      out.push(makeTurn({
        text, speaker: speakerOf(u["speaker"]),
        startMs: asMs(u["start"]), endMs: asMs(u["end"]), index: out.length,
      }));
    }
    return out;
  }
  return groupWords(data["words"] ?? [], { textKey: "text", toMs: asMs });
}

function fromDeepgram(data: Json): Turn[] {
  const results = data["results"] ?? {};
  const utterances = results["utterances"] ?? [];
  if (utterances.length) {
    const out: Turn[] = [];
    for (const u of utterances) {
      const text = String(u["transcript"] ?? "").trim();
      if (!text) continue;
      out.push(makeTurn({
        text, speaker: speakerOf(u["speaker"]),
        startMs: secToMs(u["start"]), endMs: secToMs(u["end"]), index: out.length,
      }));
    }
    return out;
  }

  const channels = results["channels"] ?? [];
  const alt = channels.length ? (channels[0]["alternatives"] ?? [{}])[0] ?? {} : {};

  const paragraphs = alt["paragraphs"]?.["paragraphs"] ?? [];
  if (paragraphs.length) {
    const out: Turn[] = [];
    for (const p of paragraphs) {
      const sentences = p["sentences"] ?? [];
      const text = sentences.map((s: Json) => String(s["text"] ?? "").trim()).join(" ").trim();
      if (!text) continue;
      out.push(makeTurn({
        text, speaker: speakerOf(p["speaker"]),
        startMs: secToMs(p["start"] ?? sentences[0]?.["start"]),
        endMs: secToMs(p["end"] ?? sentences[sentences.length - 1]?.["end"]),
        index: out.length,
      }));
    }
    return out;
  }

  return groupWords(alt["words"] ?? [], {
    textKey: "punctuated_word", fallbackKey: "word", toMs: secToMs,
  });
}

function fromRev(data: Json): Turn[] {
  const out: Turn[] = [];
  for (const mono of data["monologues"] ?? []) {
    const elements = mono["elements"] ?? [];
    const text = elements.map((e: Json) => String(e["value"] ?? "")).join("").trim();
    if (!text) continue;
    const timed = elements.filter((e: Json) => e["ts"] !== null && e["ts"] !== undefined);
    out.push(makeTurn({
      text, speaker: speakerOf(mono["speaker"]),
      startMs: timed.length ? secToMs(timed[0]["ts"]) : null,
      endMs: timed.length ? secToMs(timed[timed.length - 1]["end_ts"]) : null,
      index: out.length,
    }));
  }
  return out;
}

function fromSpeechmatics(data: Json): Turn[] {
  const words = (data["results"] ?? []).map((item: Json) => {
    const alt = (item["alternatives"] ?? [{}])[0] ?? {};
    return {
      text: alt["content"] ?? "", speaker: alt["speaker"],
      start: item["start_time"], end: item["end_time"], type: item["type"],
    };
  });
  return groupWords(words, { textKey: "text", toMs: secToMs, joinPunct: true });
}

/**
 * Best effort over an unknown but turn-shaped list. Time units are guessed
 * from magnitude: a `start` of 125000 in a file of integers is milliseconds,
 * not thirty-four hours of audio.
 */
function fromGeneric(items: Json[]): Turn[] {
  const pick = (d: Json, keys: string[]): Json => {
    for (const k of keys) {
      if (d[k] !== null && d[k] !== undefined && d[k] !== "") return d[k];
    }
    return null;
  };
  const startKeys = ["start", "start_ms", "startTime", "ts", "begin"];
  const endKeys = ["end", "end_ms", "endTime", "end_ts", "stop"];

  const numeric = items
    .filter((d) => typeof d === "object" && d !== null)
    .map((d) => pick(d, startKeys))
    .filter((v): v is number => typeof v === "number");
  const useMs =
    numeric.length > 0 &&
    Math.max(...numeric) > 10_000 &&
    numeric.every((v) => Number.isInteger(v));
  const conv = useMs ? asMs : secToMs;

  const out: Turn[] = [];
  for (const d of items) {
    if (typeof d !== "object" || d === null) continue;
    const text = String(pick(d, ["text", "transcript", "value", "content"]) ?? "").trim();
    if (!text) continue;
    out.push(makeTurn({
      text,
      speaker: speakerOf(pick(d, ["speaker", "speaker_label", "speakerId", "spk"])),
      startMs: conv(pick(d, startKeys)),
      endMs: conv(pick(d, endKeys)),
      index: out.length,
    }));
  }
  return out;
}

const DISPATCH: Record<Vendor, (d: Json) => Turn[]> = {
  whisper: fromWhisper,
  assemblyai: fromAssemblyAI,
  deepgram: fromDeepgram,
  rev: fromRev,
  speechmatics: fromSpeechmatics,
  generic_list: (d) => fromGeneric(d),
  generic_dict: (d) =>
    fromGeneric(d["utterances"] ?? d["turns"] ?? d["transcript"] ?? d["items"] ?? d["entries"] ?? []),
};

/** Parse speech-to-text vendor JSON into turns. */
export function parseAsrJson(
  text: string,
  opts: ParseOptions & { vendor?: Vendor } = {},
): Transcript {
  const { source = null, merge = true, maxGapMs = null, vendor } = opts;
  const data = JSON.parse(text.replace(/^﻿/, ""));
  const kind = vendor ?? identify(data);
  if (kind === null || !(kind in DISPATCH)) {
    throw new TypeError(
      `Unrecognised ASR JSON. Supported: ${[...VENDORS].sort().join(", ")}.`,
    );
  }
  const cues = DISPATCH[kind](data);
  const turns = merge ? mergeTurns(cues, { maxGapMs }) : cues;
  return new Transcript(turns, source, `json:${kind}`, {}, {
    vendor: kind, cue_count: cues.length,
  });
}
