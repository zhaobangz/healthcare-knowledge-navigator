"""Format-independent parser output, plus the text and table helpers every parser shares.

Every parser (HTML, DailyMed SPL, PMC JATS) produces the same shape: a list of sections,
each with a heading path and an ordered list of blocks. A block is a paragraph to be
sentence-split, or an already-atomic line (list item, table row, caption) that becomes
exactly one citable sentence. Turning table rows into self-describing lines
("Drug: ceftriaxone; Dose: 500 mg IM") matters because guideline regimens live in tables,
and a bare cell like "500 mg" is uncitable on its own.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from bs4 import Tag

BlockKind = Literal["paragraph", "list_item", "table_row", "caption"]


@dataclass
class Block:
    kind: BlockKind
    text: str
    page: int | None = None


@dataclass
class ParsedSection:
    heading_path: list[str]
    anchor: str | None = None
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ParsedDocument:
    sections: list[ParsedSection]
    title: str | None = None
    authors: str = ""
    license: str | None = None
    pub_year: int | None = None
    doc_type: str | None = None


_WS = re.compile(r"\s+")
# Brackets/parens left empty once reference markers are removed: "[ , ]", "( )".
_EMPTY_BRACKETS = re.compile(r"\[\s*(?:[,;–\-]\s*)*\]|\(\s*(?:[,;–\-]\s*)*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove the debris reference-marker stripping leaves."""
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ").replace("​", "")
    text = _WS.sub(" ", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text)
    return text.strip()


# Elements whose boundaries separate words. Inline tags (b, i, span, a…) must NOT get a
# separator, or "<i>µ</i>g/mL" would become "µ g/mL".
_BOUNDARY_TAGS = [
    "p", "div", "li", "td", "th", "tr", "caption", "dt", "dd", "h1", "h2", "h3", "h4",
    "h5", "h6", "section", "article", "blockquote", "table", "ul", "ol", "paragraph",
    "item", "title", "list", "figcaption", "label",
]  # fmt: skip


def prepare_tree(root: Tag) -> None:
    """Make word boundaries explicit once, so ``node_text`` can join strings with ''."""
    for br in tags(root.find_all("br")):
        br.replace_with(" ")
    for el in tags(root.find_all(_BOUNDARY_TAGS)):
        el.insert(0, " ")
        el.append(" ")


def node_text(node: Tag) -> str:
    return clean_text(node.get_text(""))


def tags(elements: Iterable[object]) -> list[Tag]:
    """Narrow a bs4 result set to Tags (bs4's typing also admits strings and comments)."""
    return [el for el in elements if isinstance(el, Tag)]


def _expand_grid(table: Tag) -> list[list[str]]:
    """Expand rowspan/colspan into a rectangular grid of cell texts."""
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}  # (row, col) → text carried down by rowspan
    rows = [tr for tr in tags(table.find_all("tr")) if tr.find_parent("table") is table]
    for r, tr in enumerate(rows):
        row: list[str] = []
        c = 0
        for cell in tags(tr.find_all(["td", "th"], recursive=False)):
            while (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
            text = node_text(cell)
            colspan = _int_attr(cell, "colspan")
            rowspan = _int_attr(cell, "rowspan")
            for dc in range(colspan):
                row.append(text)
                for dr in range(1, rowspan):
                    pending[(r + dr, c + dc)] = text
            c += colspan
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        grid.append(row)
    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def _int_attr(cell: Tag, name: str) -> int:
    value = cell.get(name)
    try:
        return max(1, min(int(str(value)), 50)) if value else 1
    except ValueError:
        return 1


def _header_row_count(table: Tag, grid: list[list[str]]) -> int:
    thead = table.find("thead")
    if isinstance(thead, Tag):
        return max(1, len(thead.find_all("tr")))
    first = table.find("tr")
    if isinstance(first, Tag):
        cells = tags(first.find_all(["td", "th"], recursive=False))
        if cells and all(cell.name == "th" for cell in cells):
            return 1
    # No markup hint: treat the first row as the header when it is fully populated.
    return 1 if grid and all(grid[0]) else 0


_CONNECTORS = {"plus", "or", "and", "and/or", "with", "followed by"}


def merge_connector_rows(lines: list[str]) -> list[str]:
    """Join regimen rows split by a bare connector row.

    Guideline regimen boxes put "plus" / "or" on their own row between drugs; kept
    separate, "Gentamicin 240 mg IM" would be cited without the azithromycin it must be
    given with.
    """
    out: list[str] = []
    joining = False
    for line in lines:
        if line.strip().lower() in _CONNECTORS and out:
            out[-1] = f"{out[-1]} {line.strip()}"
            joining = True
        elif joining:
            out[-1] = f"{out[-1]} {line}"
            joining = False
        else:
            out.append(line)
    return out


def table_blocks(table: Tag) -> list[Block]:
    """Render a table as a caption block plus one self-describing line per row."""
    blocks: list[Block] = []
    caption_el = table.find("caption")
    caption = node_text(caption_el) if isinstance(caption_el, Tag) else ""
    if caption:
        blocks.append(Block("caption", caption))

    grid = [row for row in _expand_grid(table) if any(row)]
    if not grid:
        return blocks
    width = len(grid[0])

    if width == 1:
        lines = [cell for (cell,) in grid if cell and cell != caption]
        blocks.extend(Block("table_row", line) for line in merge_connector_rows(lines))
        return blocks

    n_header = _header_row_count(table, grid)
    header_rows, body = grid[:n_header], grid[n_header:]
    header = (
        [
            " / ".join(dict.fromkeys(h for h in column if h))
            for column in zip(*header_rows, strict=True)
        ]
        if header_rows
        else [""] * width
    )

    for row in body:
        distinct = {cell for cell in row if cell}
        if len(distinct) == 1:  # a spanning group label or footnote row
            text = distinct.pop()
            if text != caption:
                blocks.append(Block("table_row", text))
            continue
        parts: list[str] = []
        for head, cell in zip(header, row, strict=True):
            if not cell or (parts and parts[-1].endswith(cell)):
                continue
            parts.append(f"{head}: {cell}" if head and head != cell else cell)
        if parts:
            blocks.append(Block("table_row", "; ".join(parts)))
    return blocks
