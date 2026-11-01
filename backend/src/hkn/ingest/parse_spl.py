"""DailyMed Structured Product Labeling (HL7 SPL) parser.

SPL is a nested `<section><title>…<text>…</text><component><section>…` tree. Section
titles become the heading path; `<paragraph>`, `<list><item>` and `<table>` become blocks.
The "Highlights" excerpt duplicates the full label and the package-label panel is image
text, so both are skipped.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from hkn.ingest.parse_html import ParseError
from hkn.ingest.parsed import (
    Block,
    ParsedDocument,
    ParsedSection,
    node_text,
    prepare_tree,
    table_blocks,
    tags,
)

_SKIP_TITLES = re.compile(r"package label|principal display panel", re.I)
# Package label panel. NOT 42229-5 ("SPL unclassified"): labels use it for ordinary
# subsections such as WARNINGS > Hypersensitivity.
_SKIP_CODES = {"51945-4"}


def _title_text(section: Tag) -> str:
    title = section.find("title", recursive=False)
    return node_text(title) if isinstance(title, Tag) else ""


def _text_blocks(text_el: Tag) -> list[Block]:
    blocks: list[Block] = []
    for child in tags(text_el.children):
        if child.name == "paragraph":
            blocks.append(Block("paragraph", node_text(child)))
        elif child.name == "list":
            for item in tags(child.find_all("item", recursive=False)):
                blocks.append(Block("list_item", node_text(item)))
        elif child.name == "table":
            blocks.extend(table_blocks(child))
        elif child.name not in {"renderMultiMedia", "footnote"} and node_text(child):
            blocks.extend(_text_blocks(child) or [Block("paragraph", node_text(child))])
    return [b for b in blocks if b.text]


def _walk(section: Tag, path: list[str], out: list[ParsedSection]) -> None:
    code = section.find("code", recursive=False)
    code_value = code.get("code") if isinstance(code, Tag) else None
    title = _title_text(section)
    if (title and _SKIP_TITLES.search(title)) or code_value in _SKIP_CODES:
        return
    here = [*path, title] if title else path
    anchor = section.get("ID")
    blocks: list[Block] = []
    for text_el in tags(section.find_all("text", recursive=False)):
        blocks.extend(_text_blocks(text_el))
    if blocks:
        out.append(
            ParsedSection(heading_path=here, anchor=str(anchor) if anchor else None, blocks=blocks)
        )
    for component in tags(section.find_all("component", recursive=False)):
        for child in tags(component.find_all("section", recursive=False)):
            _walk(child, here, out)


def parse_spl(xml: str) -> ParsedDocument:
    soup = BeautifulSoup(xml, "lxml-xml")
    body = soup.find("structuredBody")
    if not isinstance(body, Tag):
        raise ParseError("SPL structuredBody not found")
    for excerpt in tags(body.find_all("excerpt")):
        excerpt.decompose()
    prepare_tree(body)
    sections: list[ParsedSection] = []
    for component in tags(body.find_all("component", recursive=False)):
        for section in tags(component.find_all("section", recursive=False)):
            _walk(section, [], sections)
    if not sections:
        raise ParseError("no SPL sections extracted")
    doc_title = soup.find("title")
    return ParsedDocument(
        sections=sections, title=node_text(doc_title) if isinstance(doc_title, Tag) else None
    )
