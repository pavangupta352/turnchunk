"""The chunker.

Four rules, in the order they are applied:

1. **Oversized turns split internally at sentence boundaries**, and every piece
   keeps its speaker. A forty-minute monologue must not become one giant chunk,
   but it must also never be cut mid-sentence.
2. **A chunk boundary may only fall at a turn boundary.** This is the whole
   point of the library: a chunk never contains the tail of one person's answer
   glued to the start of another's.
3. **Overlap respects turns**, and carried-over turns are marked, so an
   aggregate over chunks does not count the same speaker twice.
4. **Short tails merge backwards.** A 40-character trailing chunk is a stub;
   stubs match everything weakly and crowd real content out of the top-k.

Sizes are measured on the *rendered* text -- the string that will actually be
embedded, speaker label included -- because that is what has to fit in the
budget. Measuring the raw text and then prepending labels is how you end up
with chunks 20% over target.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from .sentences import split_sentences
from .types import Chunk, Transcript, Turn, turns_from

SizeFn = Callable[[str], int]

DEFAULT_TEMPLATE = "{speaker}: {text}"
DEFAULT_SEPARATOR = "\n"
UNKNOWN_SPEAKER = "UNKNOWN"


def render_turn(turn: Turn, template: str = DEFAULT_TEMPLATE) -> str:
    """Render one turn to the text that will be embedded."""
    speaker = turn.speaker if turn.speaker is not None else UNKNOWN_SPEAKER
    return template.format(
        speaker=speaker,
        text=turn.text,
        start_ms=turn.start_ms if turn.start_ms is not None else "",
        end_ms=turn.end_ms if turn.end_ms is not None else "",
    )


def _interpolate(turn: Turn, lo: float, hi: float) -> Tuple[Optional[int], Optional[int]]:
    """Proportionally allocate a turn's time span across a character range.

    Used only when one turn is split into several pieces. Speech is roughly
    linear in characters-per-second, so this is a decent estimate -- but it is
    an estimate, and pieces produced this way carry ``time_estimated`` in their
    metadata so nobody mistakes it for measured data.
    """
    if turn.start_ms is None or turn.end_ms is None:
        return turn.start_ms, turn.end_ms
    span = turn.end_ms - turn.start_ms
    return (
        int(turn.start_ms + span * lo),
        int(turn.start_ms + span * hi),
    )


def split_oversized_turn(
    turn: Turn,
    target: int,
    size_fn: SizeFn,
    template: str,
) -> List[Turn]:
    """Rule 1: break a too-long turn at sentence boundaries, keeping the speaker.

    If a single sentence still exceeds the target it is hard-split at word
    boundaries -- there is no legal boundary left, and refusing to split would
    hand the caller a chunk that blows their context budget.
    """
    if size_fn(render_turn(turn, template)) <= target:
        return [turn]

    sentences = split_sentences(turn.text)
    if len(sentences) <= 1:
        sentences = _hard_split(turn.text, target, size_fn)

    # Overhead of the label, so pieces measured on raw text still fit rendered.
    overhead = size_fn(render_turn(Turn(text="", speaker=turn.speaker), template))
    budget = max(1, target - overhead)

    pieces: List[str] = []
    buf = ""
    for s in sentences:
        if buf and size_fn(buf + s) > budget:
            pieces.append(buf)
            buf = s
        else:
            buf += s
        while size_fn(buf) > budget:
            head, buf = _cut_at(buf, budget, size_fn)
            pieces.append(head)
    if buf:
        pieces.append(buf)

    total = sum(len(p) for p in pieces) or 1
    out: List[Turn] = []
    cursor = 0
    for i, p in enumerate(pieces):
        lo = cursor / total
        cursor += len(p)
        hi = cursor / total
        start_ms, end_ms = _interpolate(turn, lo, hi)
        meta = dict(turn.meta)
        meta["part"] = [i, len(pieces)]
        # The first piece keeps the turn's real start and the last its real end;
        # every boundary in between is interpolated, so mark those as estimates.
        interior = i != 0 or i != len(pieces) - 1
        if turn.start_ms is not None and len(pieces) > 1 and interior:
            meta["time_estimated"] = True
        out.append(
            Turn(
                text=p.strip() or p,
                speaker=turn.speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                index=turn.index,
                raw_speaker=turn.raw_speaker,
                meta=meta,
            )
        )
    return out


def _hard_split(text: str, target: int, size_fn: SizeFn) -> List[str]:
    """Last resort: split on whitespace when a single sentence is too long."""
    out: List[str] = []
    buf = ""
    for word in text.split(" "):
        candidate = (buf + " " + word) if buf else word
        if buf and size_fn(candidate) > target:
            out.append(buf + " ")
            buf = word
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out or [text]


def _cut_at(text: str, budget: int, size_fn: SizeFn) -> Tuple[str, str]:
    """Cut ``text`` so the head fits the budget, preferring a word boundary."""
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size_fn(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cut = lo
    space = text.rfind(" ", 0, cut)
    if space > cut * 0.6:
        cut = space + 1
    return text[:cut], text[cut:]


def chunk(
    turns: Any,
    *,
    target: int = 2000,
    overlap: int = 0,
    min_tail_ratio: float = 1 / 3,
    size_fn: SizeFn = len,
    template: str = DEFAULT_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
    source: Optional[str] = None,
    max_overflow: float = 1.5,
) -> List[Chunk]:
    """Chunk a transcript at speaker-turn boundaries.

    Args:
        turns: a :class:`Transcript`, a list of :class:`Turn`, or a list of dicts.
        target: desired size of each chunk, measured by ``size_fn`` on the
            rendered text.
        overlap: how much of the previous chunk to repeat at the start of the
            next one. Rounded up to whole turns -- overlap never cuts a turn.
        min_tail_ratio: a final chunk smaller than ``target * min_tail_ratio``
            is merged backwards instead of being emitted as a stub.
        size_fn: how to measure size. Defaults to character count; pass a
            tokeniser's counter to budget in tokens instead.
        template: how each turn is rendered into the chunk text.
        separator: what joins rendered turns.
        source: recorded on every chunk and mixed into its id.
        max_overflow: refuse a tail merge that would push a chunk beyond
            ``target * max_overflow``.

    Returns:
        A list of :class:`Chunk`, in order.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= target:
        raise ValueError("overlap must be smaller than target")
    if not 0 <= min_tail_ratio < 1:
        raise ValueError("min_tail_ratio must be in [0, 1)")

    items = turns_from(turns)
    if source is None and isinstance(turns, Transcript):
        source = turns.source
    if not items:
        return []

    # Rule 1 -- explode oversized turns first, so packing only ever sees units
    # that can fit on their own.
    units: List[Turn] = []
    for t in items:
        units.extend(split_oversized_turn(t, target, size_fn, template))

    rendered = [render_turn(u, template) for u in units]
    sizes = [size_fn(r) for r in rendered]
    sep_size = size_fn(separator)

    # Rules 2 and 3 -- greedy packing on turn boundaries, with turn-aligned overlap.
    groups: List[List[int]] = []
    current: List[int] = []
    current_size = 0

    for i in range(len(units)):
        addition = sizes[i] + (sep_size if current else 0)
        if current and current_size + addition > target:
            groups.append(current)
            carried = _carry(current, sizes, sep_size, overlap, target)
            current = list(carried)
            current_size = _measure(current, sizes, sep_size)
            addition = sizes[i] + (sep_size if current else 0)
        current.append(i)
        current_size += addition
    if current:
        groups.append(current)

    chunks = [
        _build(g, units, rendered, separator, source, overlap_count=_overlap_count(g, groups, gi))
        for gi, g in enumerate(groups)
    ]

    # Rule 4 -- merge a stub tail backwards.
    chunks = _merge_tail(
        chunks, groups, units, rendered, separator, source,
        target=target, min_tail_ratio=min_tail_ratio,
        size_fn=size_fn, max_overflow=max_overflow,
    )
    return chunks


def _measure(indices: Sequence[int], sizes: Sequence[int], sep_size: int) -> int:
    if not indices:
        return 0
    return sum(sizes[i] for i in indices) + sep_size * (len(indices) - 1)


def _carry(
    emitted: Sequence[int],
    sizes: Sequence[int],
    sep_size: int,
    overlap: int,
    target: int,
) -> List[int]:
    """Choose whole turns from the end of ``emitted`` to repeat as overlap.

    Never carries the entire chunk -- that would make no forward progress and
    loop forever -- and never carries more than half the target, which would
    leave no room for new content.
    """
    if overlap <= 0 or len(emitted) <= 1:
        return []
    cap = min(overlap, target // 2)
    carried: List[int] = []
    total = 0
    for i in reversed(emitted[1:]):  # always leave at least the first turn behind
        addition = sizes[i] + (sep_size if carried else 0)
        if total + addition > cap and carried:
            break
        carried.insert(0, i)
        total += addition
        if total >= cap:
            break
    return carried


def _overlap_count(group: Sequence[int], groups: Sequence[Sequence[int]], gi: int) -> int:
    """How many leading turns of this group were carried from the previous one."""
    if gi == 0:
        return 0
    prev = set(groups[gi - 1])
    n = 0
    for i in group:
        if i in prev:
            n += 1
        else:
            break
    return n


def _build(
    group: Sequence[int],
    units: Sequence[Turn],
    rendered: Sequence[str],
    separator: str,
    source: Optional[str],
    overlap_count: int,
) -> Chunk:
    return Chunk(
        text=separator.join(rendered[i] for i in group),
        turns=[units[i] for i in group],
        overlap_indices=list(range(overlap_count)),
        source=source,
    )


def _merge_tail(
    chunks: List[Chunk],
    groups: List[List[int]],
    units: Sequence[Turn],
    rendered: Sequence[str],
    separator: str,
    source: Optional[str],
    *,
    target: int,
    min_tail_ratio: float,
    size_fn: SizeFn,
    max_overflow: float,
) -> List[Chunk]:
    if len(chunks) < 2 or min_tail_ratio <= 0:
        return chunks

    last_group = groups[-1]
    overlap_n = len(chunks[-1].overlap_indices)
    own = last_group[overlap_n:]
    if not own:
        # The tail is nothing but overlap -- it carries no new content at all.
        return chunks[:-1]

    own_size = size_fn(separator.join(rendered[i] for i in own))
    if own_size >= target * min_tail_ratio:
        return chunks

    merged_group = groups[-2] + own
    merged_text = separator.join(rendered[i] for i in merged_group)
    if size_fn(merged_text) > target * max_overflow:
        return chunks

    groups[-2] = merged_group
    chunks[-2] = _build(
        merged_group, units, rendered, separator, source,
        overlap_count=len(chunks[-2].overlap_indices),
    )
    return chunks[:-1]
