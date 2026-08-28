"""Framework adapters.

They must import without the framework installed, and fail with a clear,
actionable message only when actually used.
"""

from __future__ import annotations

import pytest

from turnchunk.integrations.chonkie import ConversationChunker
from turnchunk.integrations.langchain import TurnChunkSplitter, chunk_to_metadata
from turnchunk.integrations.llamaindex import TurnChunkNodeParser

EXAMPLE = "examples/renewal-call.vtt"


def test_chonkie_adapter_needs_no_third_party_install():
    chunks = ConversationChunker(chunk_size=400)(EXAMPLE)
    assert chunks and chunks[0].primary_speaker == "Priya Raman"
    assert chunks[0].start_ms is not None


def test_chonkie_adapter_batches():
    out = ConversationChunker(chunk_size=400)([EXAMPLE, EXAMPLE])
    assert len(out) == 2 and all(o for o in out)


def test_metadata_contract_is_stable():
    """Downstream filters depend on these keys; they are part of the API."""
    c = ConversationChunker(chunk_size=400).chunk(EXAMPLE)[0]
    meta = chunk_to_metadata(c)
    assert {"chunk_id", "source", "speaker", "speakers",
            "start_ms", "end_ms", "turn_start", "turn_end"} == set(meta)


def test_adapters_import_without_their_framework():
    TurnChunkSplitter(chunk_size=100)
    TurnChunkNodeParser(chunk_size=100)


def test_langchain_adapter_errors_clearly_when_unavailable():
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="LangChain is not installed"):
            TurnChunkSplitter().split_transcript(EXAMPLE)
    else:
        docs = TurnChunkSplitter(chunk_size=400).split_transcript(EXAMPLE)
        assert docs[0].metadata["speaker"] == "Priya Raman"


def test_llamaindex_adapter_errors_clearly_when_unavailable():
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="LlamaIndex is not installed"):
            TurnChunkNodeParser().parse_transcript(EXAMPLE)
    else:
        nodes = TurnChunkNodeParser(chunk_size=400).parse_transcript(EXAMPLE)
        assert nodes[0].metadata["speaker"] == "Priya Raman"
