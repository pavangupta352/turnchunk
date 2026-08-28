"""LangChain adapter.

    from turnchunk.integrations.langchain import TurnChunkSplitter

    docs = TurnChunkSplitter(chunk_size=2000).split_transcript("meeting.vtt")

The result is a list of ``langchain_core.documents.Document`` whose metadata
carries the speaker, the time span and the turn range, so a retrieved document
can be attributed and opened at the moment it was said.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from ..chunker import chunk as _chunk
from ..parsers import parse as _parse
from ..speakers import resolve_speakers


def _Document():
    try:
        from langchain_core.documents import Document  # type: ignore
    except ImportError:  # pragma: no cover - exercised only without langchain
        try:
            from langchain.schema import Document  # type: ignore
        except ImportError:
            raise ImportError(
                "LangChain is not installed. `pip install langchain-core` to use "
                "this adapter, or use turnchunk.chunk() directly."
            ) from None
    return Document


def chunk_to_metadata(c) -> dict:
    return {
        "chunk_id": c.id,
        "source": c.source,
        "speaker": c.primary_speaker,
        "speakers": c.speakers,
        "start_ms": c.start_ms,
        "end_ms": c.end_ms,
        "turn_start": c.turn_start,
        "turn_end": c.turn_end,
    }


class TurnChunkSplitter:
    """A LangChain-shaped splitter that never cuts a speaker turn.

    Implements ``split_text`` and ``create_documents`` so it can stand in for a
    ``TextSplitter``, plus ``split_transcript`` which is the one you want --
    it parses the file first, so speaker and timestamp metadata survive.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        *,
        min_tail_ratio: float = 1 / 3,
        length_function=len,
        resolve_speaker_names: bool = True,
        **chunk_kwargs: Any,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tail_ratio = min_tail_ratio
        self.length_function = length_function
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
            size_fn=self.length_function,
            **self.chunk_kwargs,
        )

    def split_transcript(self, source: Any, *, format: Optional[str] = None) -> List[Any]:
        """Parse and chunk a transcript into Documents with full metadata."""
        Document = _Document()
        return [
            Document(page_content=c.text, metadata=chunk_to_metadata(c))
            for c in self._chunks(source, format)
        ]

    def split_text(self, text: str) -> List[str]:
        """TextSplitter interface. Prefer ``split_transcript`` -- this drops metadata."""
        return [c.text for c in self._chunks(text)]

    def create_documents(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[Any]:
        Document = _Document()
        out: List[Any] = []
        metadatas = list(metadatas or [])
        for i, text in enumerate(texts):
            base = dict(metadatas[i]) if i < len(metadatas) else {}
            for c in self._chunks(text):
                meta = dict(base)
                meta.update(chunk_to_metadata(c))
                out.append(Document(page_content=c.text, metadata=meta))
        return out

    def split_documents(self, documents: Iterable[Any]) -> List[Any]:
        return self.create_documents(
            [d.page_content for d in documents],
            [getattr(d, "metadata", {}) for d in documents],
        )
