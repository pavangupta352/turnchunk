"""A recursive character splitter, for comparison only.

This reimplements what LangChain's ``RecursiveCharacterTextSplitter`` and most
hand-rolled splitters do: try progressively finer separators until the text
fits. It exists so ``turnchunk viz --compare`` can show, on the reader's own
file, exactly what a general-purpose splitter does to a conversation.

It is not exported from the package root and is not for production use.
"""

from __future__ import annotations

from typing import List, Sequence

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", ", ", " ", "")


def recursive_split(
    text: str,
    target: int = 2000,
    overlap: int = 0,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> List[str]:
    chunks = _split(text, target, list(separators))
    if overlap <= 0:
        return chunks
    out: List[str] = []
    for i, c in enumerate(chunks):
        if i and overlap:
            out.append(chunks[i - 1][-overlap:] + c)
        else:
            out.append(c)
    return out


def _split(text: str, target: int, separators: List[str]) -> List[str]:
    if len(text) <= target:
        return [text] if text else []
    sep = separators[0] if separators else ""
    rest = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep) if sep else list(text)
    out: List[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= target:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > target:
            out.extend(_split(part, target, rest) if rest else [part])
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)
    return [c for c in out if c]
