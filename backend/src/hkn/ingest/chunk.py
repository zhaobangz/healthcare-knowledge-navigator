"""Chunking: runs of consecutive sentences packed to a token budget (concepts 1–2).

Chunks are the retrieval unit (small, so a match is specific); sections are the
generation unit (big, so the model sees context) — see `retrieval/expand.py` for the
small-to-big step. A chunk never crosses a section boundary, so its heading path is
unambiguous, and overlap is measured in whole sentences, so no idea is cut mid-sentence.
"""

from __future__ import annotations

from dataclasses import dataclass


def approx_tokens(text: str) -> int:
    """~4 characters per token for English; close enough for packing decisions."""
    return max(1, round(len(text) / 4))


@dataclass(frozen=True)
class ChunkSpan:
    first: int  # sentence ord, inclusive
    last: int  # sentence ord, inclusive


def pack_chunks(sentences: list[str], *, target_tokens: int, overlap: int) -> list[ChunkSpan]:
    """Greedy packing: fill each chunk up to `target_tokens`, then start the next chunk
    `overlap` sentences before the previous one ended.

    A single sentence longer than the budget becomes its own chunk rather than being
    split — splitting would create text that is not a citable unit.
    """
    if not sentences:
        return []
    sizes = [approx_tokens(s) for s in sentences]
    spans: list[ChunkSpan] = []
    start = 0
    n = len(sentences)
    while start < n:
        end = start
        total = sizes[start]
        while end + 1 < n and total + sizes[end + 1] <= target_tokens:
            end += 1
            total += sizes[end]
        spans.append(ChunkSpan(start, end))
        if end == n - 1:
            break
        start = max(start + 1, end + 1 - overlap)
    return spans


def heading_prefix(org: str, title: str, heading_path: list[str]) -> str:
    """The free context prefix (concept 13 baseline): where in which document a chunk sits."""
    path = " › ".join(heading_path)
    return f"{org} — {title}" + (f" › {path}" if path else "")
