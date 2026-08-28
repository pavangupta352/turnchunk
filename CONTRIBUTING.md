# Contributing

## The most valuable contribution

**A transcript file that breaks the parser.**

turnchunk claims to read anything. That claim is only as good as the files it
has actually seen, and hand-written fixtures are exactly the trap that makes a
"works with anything" library break on first contact with reality.

If you have a real export — Teams, Zoom, Otter, Fathom, Granola, Descript,
Rev, Sonix, a vendor nobody here has heard of — and turnchunk mangles it:

1. Open an issue with the smallest excerpt that reproduces the problem.
   **Redact freely.** Replace names with "Alice"/"Bob" and content with
   nonsense — the structure is what matters, not the words.
2. Or send a PR adding it to `tests/fixtures/` with a test in
   `tests/test_parsers.py` asserting the correct turns.

Formats already covered are listed in `turnchunk formats`.

## Development

```bash
git clone https://github.com/pavangupta352/turnchunk
cd turnchunk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

**Zero runtime dependencies.** This is a feature, not an accident. It is why
turnchunk installs in under a second and runs in a Lambda, on a Pi, and inside
an air-gapped network. A PR that adds a runtime dependency needs to argue why
the feature is worth losing that.

**Scope: transcript in, chunks out.** Parsing, turn construction, chunking, and
rendering for a prompt. No transcription, no embeddings, no storage, no
retrieval, no model calls. Features that need a model are out — that boundary is
what keeps this composable with every pipeline instead of competing with them.

**Never guess about people.** Any change to speaker resolution must keep this
property: when a merge is ambiguous, the labels stay separate. Misattributing
a statement to the wrong person is the one failure this library must not
produce. There is a test for it; do not weaken it.

**Timestamps that aren't known stay `None`.** Never `0`. Callers have to be able
to tell "at the start" from "we don't know".

## The two implementations must stay identical

`js/` is a port of the Python package, and they are held together by
`tests/corpus/conformance.json` -- generated from Python, checked in, and
asserted against by both test suites down to the SHA-256 chunk ids.

If you change behaviour in one language, you must change it in the other:

```bash
python scripts/build_corpus.py    # regenerate after a deliberate change
pytest                            # Python must match
cd js && npm test                 # TypeScript must match
```

CI fails if regenerating produces a diff, because that means the two languages
have silently drifted. Review any corpus diff carefully -- an unexplained one is
a behaviour change nobody intended.

## Tests

New behaviour needs a test. New format support needs a fixture.

The property test `test_a_turn_is_never_split` is the library's central claim.
If a change makes it fail, the change is wrong — not the test.

```bash
pytest                     # everything
pytest tests/test_chunker.py -k never_split -v
```
