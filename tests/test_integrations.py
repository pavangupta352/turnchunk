"""Framework adapters.

They must import without the framework installed, and fail with a clear,
actionable message only when actually used.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from turnchunk.integrations.chonkie import ConversationChunker
from turnchunk.integrations.langchain import TurnChunkSplitter, chunk_to_metadata
from turnchunk.integrations.llamaindex import TurnChunkNodeParser

EXAMPLE = "examples/renewal-call.vtt"


def test_chonkie_adapter_needs_no_third_party_install():
    chunks = ConversationChunker(chunk_size=400).chunk_file(EXAMPLE)
    assert chunks and chunks[0].primary_speaker == "Priya Raman"
    assert chunks[0].start_ms is not None


def test_chonkie_adapter_batches():
    out = ConversationChunker(chunk_size=400).chunk_batch([Path(EXAMPLE), Path(EXAMPLE)])
    assert len(out) == 2 and all(o for o in out)


def test_metadata_contract_is_stable():
    """Downstream filters depend on these keys; they are part of the API."""
    c = ConversationChunker(chunk_size=400).chunk_file(EXAMPLE)[0]
    meta = chunk_to_metadata(c)
    assert {"chunk_id", "source", "speaker", "speakers",
            "start_ms", "end_ms", "turn_start", "turn_end"} == set(meta)


def test_adapters_import_without_their_framework():
    TurnChunkSplitter(chunk_size=100)
    TurnChunkNodeParser(chunk_size=100)


def _importable(module: str) -> bool:
    """Can this framework actually be imported here?

    Not just "is it installed": llama-index-core is installable on Python 3.9
    but raises TypeError on import, because a transitive dependency uses PEP 604
    syntax. Any failure means unavailable.
    """
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def test_langchain_adapter_produces_real_documents_or_a_clean_error():
    if _importable("langchain_core"):
        docs = TurnChunkSplitter(chunk_size=400).split_transcript_file(EXAMPLE)
        assert type(docs[0]).__name__ == "Document"
        assert docs[0].metadata["speaker"] == "Priya Raman"
        assert docs[0].metadata["start_ms"] is not None
    else:
        with pytest.raises(ImportError, match="LangChain is not installed"):
            TurnChunkSplitter().split_transcript_file(EXAMPLE)


def test_llamaindex_adapter_produces_real_nodes_or_a_clean_error():
    if _importable("llama_index.core"):
        nodes = TurnChunkNodeParser(chunk_size=400).parse_transcript_file(EXAMPLE)
        assert type(nodes[0]).__name__ == "TextNode"
        assert nodes[0].metadata["speaker"] == "Priya Raman"
    else:
        # Whatever the framework raises, the adapter must surface ImportError.
        with pytest.raises(ImportError, match="LlamaIndex is not installed"):
            TurnChunkNodeParser().parse_transcript_file(EXAMPLE)
