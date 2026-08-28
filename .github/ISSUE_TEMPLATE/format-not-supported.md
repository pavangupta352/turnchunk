---
name: A transcript format doesn't parse
about: The most useful issue you can file
title: "[format] "
labels: format
---

**Where is the file from?**
(Teams, Zoom, Otter, Fathom, Granola, Descript, Rev, Sonix, a custom pipeline...)

**A small excerpt that reproduces it**

Please redact — replace real names with Alice/Bob and the content with anything.
The structure is what matters.

```
paste 5-20 lines here
```

**What turnchunk did**

```
$ turnchunk detect yourfile.vtt
$ turnchunk stats yourfile.vtt
```

**What it should have produced**
(how many turns, which speakers, roughly what timestamps)
