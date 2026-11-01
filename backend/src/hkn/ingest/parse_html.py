"""Generic heading-aware HTML parser, configured per source in seed_manifest.yaml.

Guideline sites differ in chrome but share a shape: a content container, a heading
hierarchy, paragraphs, lists and regimen tables. One document-order walk over the
container handles all of them; the per-source config only says where the content is,
where it starts and stops, and which reference-marker style to strip. Content before
`start_heading` and after a `stop_headings` match is discarded, so navigation, summaries
and reference lists never become citable text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from hkn.ingest.parsed import (
    Block,
    ParsedDocument,
    ParsedSection,
    node_text,
    prepare_tree,
    table_blocks,
    tags,
)

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
PARAGRAPH_TAGS = {"p", "blockquote", "pre", "dd", "dt", "figcaption", "address"}
LIST_TAGS = {"ul", "ol"}
DROP_TAGS = {
    "script", "style", "noscript", "nav", "button", "form", "svg", "img", "iframe",
    "select", "input", "template", "picture", "video", "audio", "map", "object",
}  # fmt: skip
BLOCK_TAGS = HEADINGS | PARAGRAPH_TAGS | LIST_TAGS | {"table", "li", "div", "section"}
DEFAULT_DROP_SELECTORS = [".sr-only", ".visually-hidden", "[aria-hidden=true]"]

_NUMERIC_REF = re.compile(r"^\s*\d{1,4}(?:\s*[–\-,]\s*\d{1,4})*\s*$")
# Interactive-widget text that survives as inline strings on some sites.
_JUNK_BLOCK = re.compile(r"^(expand all|collapse all|back to top|top of page|print|share)$", re.I)
_JUNK_PREFIX = re.compile(r"^(enlarge)\s+", re.I)
# A standalone evidence label qualifies the block before it (NCI summaries).
_QUALIFIER_BLOCK = re.compile(r"^\[?level of evidence [^\]]{1,30}\]?\.?$", re.I)


class ParseError(ValueError):
    """The configured structure was not found — fail loudly instead of ingesting chrome."""


@dataclass
class HtmlParserConfig:
    root: str = "main"
    start_heading: str | None = None
    stop_headings: list[str] = field(default_factory=list)
    skip_headings: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)
    strip_numeric_italics: bool = False

    @classmethod
    def from_manifest(cls, raw: dict) -> HtmlParserConfig:
        return cls(
            root=raw.get("root", "main"),
            start_heading=raw.get("start_heading"),
            stop_headings=list(raw.get("stop_headings", [])),
            skip_headings=list(raw.get("skip_headings", [])),
            drop=list(raw.get("drop", [])),
            strip_numeric_italics=bool(raw.get("strip_numeric_italics", False)),
        )


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def strip_reference_markers(root: Tag, *, numeric_italics: bool) -> None:
    """Remove citation markers so they don't pollute sentences or embeddings.

    Handles the three styles in the seed corpus: superscript numbers (USPSTF), bracketed
    anchor links to a reference list (NCI: ``[<a href="#cit/...">23</a>]``), and MMWR's
    italic numbers in parentheses (``(<i>880</i>,<i>881</i>)``). The brackets and commas
    they leave behind are removed later by ``clean_text``.
    """
    for sup in root.find_all("sup"):
        if _NUMERIC_REF.match(sup.get_text()):
            sup.decompose()
    for a in root.select('a[href^="#cit"], a[href^="#r"], a[href^="#R"]'):
        if _NUMERIC_REF.match(a.get_text()):
            a.decompose()
    if numeric_italics:
        for el in root.find_all(["i", "em"]):
            if _NUMERIC_REF.match(el.get_text()):
                el.decompose()


def _heading_anchor(heading: Tag) -> str | None:
    if heading.get("id"):
        return str(heading["id"])
    inner = heading.find(id=True) or heading.find("a", attrs={"name": True})
    if isinstance(inner, Tag):
        return str(inner.get("id") or inner.get("name"))
    return None


class _Walker:
    def __init__(self, config: HtmlParserConfig) -> None:
        self.config = config
        self.started = config.start_heading is None
        self.stopped = False
        self.skip_level: int | None = None
        self.stack: list[tuple[int, str]] = []
        self.sections: list[ParsedSection] = []
        self.current: ParsedSection | None = None

    # -- events ---------------------------------------------------------------------------
    def heading(self, el: Tag) -> None:
        text = node_text(el)
        if not text:
            return
        level = int(el.name[1])
        if not self.started:
            if self.config.start_heading and re.search(self.config.start_heading, text, re.I):
                self.started = True
            else:
                return
        if _matches(self.config.stop_headings, text):
            self.stopped = True
            return
        if self.skip_level is not None:
            if level > self.skip_level:
                return
            self.skip_level = None
        if _matches(self.config.skip_headings, text):
            self.skip_level = level
            return
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        self.current = ParsedSection(
            heading_path=[t for _, t in self.stack], anchor=_heading_anchor(el)
        )
        self.sections.append(self.current)

    def block(self, block: Block) -> None:
        if not self.started or self.stopped or self.skip_level is not None:
            return
        block.text = _JUNK_PREFIX.sub("", block.text)
        if not block.text or _JUNK_BLOCK.match(block.text):
            return
        if self.current is None:  # content before the first heading
            self.current = ParsedSection(heading_path=[])
            self.sections.append(self.current)
        if _QUALIFIER_BLOCK.match(block.text) and self.current.blocks:
            prev = self.current.blocks[-1]
            prev.text = f"{prev.text} [{block.text.strip('[].')}]"
            return
        # Responsive pages often render the same table twice (desktop + mobile).
        if any(b.text == block.text for b in self.current.blocks):
            return
        self.current.blocks.append(block)

    # -- traversal ------------------------------------------------------------------------
    def walk(self, node: Tag) -> None:
        for child in node.children:
            if self.stopped:
                return
            if not isinstance(child, Tag) or child.name in DROP_TAGS:
                continue
            name = child.name
            if name in HEADINGS:
                self.heading(child)
            elif name == "table":
                for b in table_blocks(child):
                    self.block(b)
            elif name in LIST_TAGS:
                self.list_items(child)
            elif name in PARAGRAPH_TAGS:
                self.block(Block("paragraph", node_text(child)))
            elif child.find(BLOCK_TAGS):
                self.walk(child)
            else:  # a leaf container (div/span) holding inline text only
                self.block(Block("paragraph", node_text(child)))

    def list_items(self, lst: Tag) -> None:
        for li in tags(lst.find_all("li", recursive=False)):
            nested = tags(li.find_all(list(LIST_TAGS | {"table"}), recursive=False))
            for n in nested:
                n.extract()
            if li.find(list(PARAGRAPH_TAGS | {"div"})):
                self.walk(li)  # rich list item: walk its paragraphs as list items
            else:
                self.block(Block("list_item", node_text(li)))
            for n in nested:
                if n.name == "table":
                    for b in table_blocks(n):
                        self.block(b)
                else:
                    self.list_items(n)


def parse_html(html: str, config: HtmlParserConfig) -> ParsedDocument:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(config.root)
    if root is None:
        raise ParseError(f"content root {config.root!r} not found")
    for selector in DEFAULT_DROP_SELECTORS + config.drop:
        for el in root.select(selector):
            el.decompose()
    strip_reference_markers(root, numeric_italics=config.strip_numeric_italics)
    prepare_tree(root)

    walker = _Walker(config)
    walker.walk(root)
    if not walker.started:
        raise ParseError(f"start heading {config.start_heading!r} not found")
    sections = [s for s in walker.sections if s.blocks]
    if not sections:
        raise ParseError("no content extracted")
    title_el = soup.find("h1")
    return ParsedDocument(
        sections=sections, title=node_text(title_el) if isinstance(title_el, Tag) else None
    )
