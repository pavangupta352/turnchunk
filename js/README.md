# turnchunk

**Chunking that understands who is speaking.** Parse any transcript, chunk it
without ever splitting a speaker turn, and keep the speaker and timestamp on
every chunk.

Zero dependencies. ESM + CJS + types. Runs on Node 18+, browsers, Deno, Bun and
edge runtimes — there is no `node:crypto`, no filesystem access and no
platform-specific code anywhere in it.

```bash
npm install turnchunk
```

```ts
import { parse, chunk } from "turnchunk";

const turns  = parse(vttText);       // format auto-detected from content
const chunks = chunk(turns, { target: 2000, overlap: 200 });

for (const c of chunks) {
  console.log(c.primarySpeaker, c.startMs, c.text.slice(0, 80));
}
```

Every chunk carries what you need to cite it:

```ts
c.id               // stable, content-addressed - usable as a vector-store key
c.speakers         // ["Dana Okafor"]
c.primarySpeaker   // "Dana Okafor"  (overlap excluded)
c.startMs          // 41600   -> jump a player to the moment it was said
c.turnStart        // 5       -> index back into the transcript
c.overlapIndices   // [0]     -> turns carried from the previous chunk
```

Reads WebVTT (Teams `<v>` spans, Zoom inline names, YouTube rolling captions),
SubRip, six plain-text conventions, and JSON from Whisper/WhisperX, Deepgram,
AssemblyAI, Rev.ai and Speechmatics.

`parse()` takes a string rather than a path, because the same build has to run
where there is no filesystem. Read the file yourself and pass the contents.

This package is byte-for-byte identical to the Python package of the same name,
verified in CI against a shared conformance corpus — including the chunk ids.

**Full documentation: https://github.com/pavangupta352/turnchunk**

MIT.
