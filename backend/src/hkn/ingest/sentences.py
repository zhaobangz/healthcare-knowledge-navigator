"""Sentence segmentation — the citation unit.

Sentences are what Claude cites (one search-result text block each), so a bad split is a
user-visible defect: a dose separated from its drug, or "Sect." ending a sentence. pysbd
handles decimals, "e.g." and "vs." well; `_repair` fixes the residual cases it gets
wrong on clinical text (unit and section abbreviations, fragments starting lowercase).

List items, table rows and captions are already atomic and become exactly one
sentence each, unless a list item itself contains several sentences.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache

from hkn.ingest.parsed import Block, ParsedSection

with warnings.catch_warnings():  # pysbd ships regexes with invalid escapes
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

# Abbreviations after which pysbd sometimes ends a sentence on clinical text.
_ABBREV_END = re.compile(
    r"\b(?:sect|secs?|yrs?|wks?|mos?|hrs?|mins?|approx|ref|refs|tab|tabs|fig|figs|no|nos|"
    r"vol|cf|incl|dept|ca|est|max|min|resp|ed|eds|pt|pts|dx|tx|hx|vs|e\.g|i\.e|al)\.\s*$",
    re.IGNORECASE,
)
_NUMERIC_FRAGMENT = re.compile(r"^[\d.\-–,;:()%\s]+$")
# Trailing qualifiers such as NCI's "[Level of evidence B1]" belong to the sentence before.
_BRACKET_QUALIFIER = re.compile(r"^\[[^\]]{1,60}\]\.?$")
# No "^": used with Pattern.match(text, pos), which anchors at pos ("^" would not).
_LEADING_BRACKET = re.compile(r"\s*\[[^\]]{1,60}\]\.?\s*")

SentenceKind = str  # "sentence" | "list_item" | "table_row" | "caption"


@dataclass
class SentenceRecord:
    text: str
    kind: SentenceKind
    char_start: int  # offsets into the section text built by `build_section_text`
    char_end: int
    page: int | None = None


@lru_cache(maxsize=1)
def _segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def _repair(spans: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in spans:
        piece = text[start:end].strip()
        if not piece:
            continue
        lead = _LEADING_BRACKET.match(text, start, end) if out else None
        if lead and lead.end() < end:
            # "…benefit.[Level of evidence A1] Next sentence" → qualifier joins the previous.
            out[-1] = (out[-1][0], lead.end())
            start = lead.end()
            piece = text[start:end].strip()
        if out:
            prev = text[out[-1][0] : out[-1][1]]
            if (
                _ABBREV_END.search(prev)
                or piece[0].islower()
                or _NUMERIC_FRAGMENT.match(piece)
                or _BRACKET_QUALIFIER.match(piece)
                or len(piece) < 4
            ):
                out[-1] = (out[-1][0], end)
                continue
        out.append((start, end))
    return out


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences in `text`, trailing whitespace excluded."""
    if not text.strip():
        return []
    # char_span=True makes pysbd return TextSpan objects (its annotations say str).
    spans = [(span.start, span.end) for span in _segmenter().segment(text)]  # type: ignore[union-attr]
    # pysbd's char spans include trailing whitespace; trim so highlights are tight.
    trimmed = []
    for start, end in _repair(spans, text):
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end and text[start].isspace():
            start += 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def build_section_text(section: ParsedSection) -> tuple[str, list[SentenceRecord]]:
    """Concatenate a section's blocks and locate every sentence in the result.

    Blocks are joined with newlines; a paragraph keeps its original inner spacing, so
    `section_text[s.char_start:s.char_end] == s.text` holds for every sentence.
    """
    parts: list[str] = []
    records: list[SentenceRecord] = []
    offset = 0
    for block in section.blocks:
        text = block.text
        spans = _block_spans(block)
        for start, end in spans:
            kind = "sentence" if block.kind == "paragraph" else block.kind
            records.append(
                SentenceRecord(
                    text=text[start:end],
                    kind=kind,
                    char_start=offset + start,
                    char_end=offset + end,
                    page=block.page,
                )
            )
        parts.append(text)
        offset += len(text) + 1  # the joining "\n"
    return "\n".join(parts), records


def _block_spans(block: Block) -> list[tuple[int, int]]:
    if block.kind in {"table_row", "caption"}:
        return [(0, len(block.text))]
    return split_sentences(block.text)
