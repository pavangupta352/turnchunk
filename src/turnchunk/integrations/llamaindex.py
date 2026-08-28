"""LlamaIndex adapter.

    from turnchunk.integrations.llamaindex import TurnChunkNodeParser

    nodes = TurnChunkNodeParser(chunk_size=2000).parse_transcript("meeting.vtt")
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..chunker import chunk as _chunk
from ..parsers import parse as _parse
from ..speakers import resolve_speakers
from .langchain import chunk_to_metadata


def _TextNode():
    try:
        from llama_index.core.schema import TextNode  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from llama_index.schema import TextNode  # type: ignore
        except Exception as exc:
            raise ImportError(
                "LlamaIndex is not installed. `pip install llama-index-core` to use "
                "this adapter, or use turnchunk.chunk() directly."
            ) from exc
    return TextNode


class TurnChunkNodeParser:
    """Turn transcripts into TextNodes without ever splitting a speaker turn."""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        *,
        min_tail_ratio: float = 1 / 3,
        resolve_speaker_names: bool = True,
        **chunk_kwargs: Any,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tail_ratio = min_tail_ratio
        self.resolve_speaker_names = resolve_speaker_names
        self.chunk_kwargs = chunk_kwargs

    def _chunks(self, source: Any, fmt: Optional[str] = None):
        transcript = _parse(source, format=fmt)
        if self.resolve_speaker_names:
            resolve_speakers(transcript.turns)
        return _chunk(
            transcript,
            target=self.chunk_size,
            overlap=self.chunk_overlap,
            min_tail_ratio=self.min_tail_ratio,
            **self.chunk_kwargs,
        )

    def parse_transcript(self, source: Any, *, format: Optional[str] = None) -> List[Any]:
        TextNode = _TextNode()
        return [
            TextNode(text=c.text, id_=c.id, metadata=chunk_to_metadata(c))
            for c in self._chunks(source, format)
        ]

    def get_nodes_from_documents(
        self, documents: Sequence[Any], show_progress: bool = False, **kwargs: Any
    ) -> List[Any]:
        TextNode = _TextNode()
        out: List[Any] = []
        for doc in documents:
            text = getattr(doc, "text", None) or getattr(doc, "get_content", lambda: "")()
            base = dict(getattr(doc, "metadata", {}) or {})
            for c in self._chunks(text):
                meta = dict(base)
                meta.update(chunk_to_metadata(c))
                out.append(TextNode(text=c.text, id_=c.id, metadata=meta))
        return out

    # LlamaIndex calls node parsers directly in some pipelines.
    def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> List[Any]:
        return self.get_nodes_from_documents(nodes, **kwargs)
