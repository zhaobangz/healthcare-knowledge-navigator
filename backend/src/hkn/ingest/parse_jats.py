"""PubMed Central JATS XML parser.

Only CC BY and CC0 articles are admitted (`classify_license`); anything else — including
CC BY-NC/ND, publisher-specific licenses, or a missing `<license>` — is rejected before
any text is stored, so the corpus never holds text it may not redistribute as excerpts.
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

ARTICLE_TYPE_TO_DOC_TYPE = {
    "research-article": "other",  # refined below from title/abstract cues
    "review-article": "review",
    "systematic-review": "systematic_review",
    "meta-analysis": "systematic_review",
    "practice-guideline": "guideline",
    "guideline": "guideline",
}
_RCT_CUE = re.compile(r"\brandomi[sz]ed\b.*\b(trial|controlled)\b", re.I)
_SR_CUE = re.compile(r"\b(systematic review|meta-analys[ie]s)\b", re.I)


class LicenseRejected(ValueError):
    pass


def classify_license(text: str) -> str | None:
    """Return 'cc-by' / 'cc0' for admissible licenses, else None.

    Checks the license URL first (most precise), then the human-readable text. Any
    NC/ND/SA modifier makes the license inadmissible.
    """
    t = text.lower()
    if re.search(r"publicdomain/zero|\bcc0\b|cc-zero", t):
        return "cc0"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/", t)
    if m:
        return "cc-by" if m.group(1) == "by" else None
    if re.search(r"\b(non-?commercial|no-?deriv|share-?alike|by-nc|by-nd|by-sa)\b", t):
        return None
    if re.search(
        r"creative commons attribution(?: 4\.0| 3\.0| 2\.0)?(?: international)? licen[cs]e|\bcc[ -]by\b",
        t,
    ):
        return "cc-by"
    return None


def _license(soup: BeautifulSoup) -> tuple[str | None, str]:
    lic = soup.find("license")
    if not isinstance(lic, Tag):
        return None, ""
    # Prefer the machine-readable reference (ali:license_ref, or xlink:href on <license>).
    ref_el = lic.find("license_ref") or lic.find("ali:license_ref")
    ref = node_text(ref_el) if isinstance(ref_el, Tag) else ""
    href = ref or str(lic.get("xlink:href") or lic.get("href") or "")
    text = node_text(lic)
    return classify_license(f"{href} {text}"), (href or text)[:300]


def _authors(soup: BeautifulSoup) -> str:
    """Vancouver style: "Lee A, Smith JB, Chen Y, et al." """
    names: list[str] = []
    for contrib in tags(soup.select('contrib[contrib-type="author"]')):
        surname = contrib.find("surname")
        given = contrib.find("given-names")
        if isinstance(surname, Tag):
            initials = (
                "".join(p[0] for p in node_text(given).replace("-", " ").split())
                if isinstance(given, Tag)
                else ""
            )
            names.append(f"{node_text(surname)} {initials}".strip())
    if len(names) > 3:
        return ", ".join(names[:3]) + ", et al."
    return ", ".join(names)


def journal_title(soup: BeautifulSoup) -> str:
    el = soup.find("journal-title")
    return node_text(el) if isinstance(el, Tag) else ""


def pmcid(soup: BeautifulSoup) -> str | None:
    for el in tags(soup.find_all("article-id")):
        if el.get("pub-id-type") in {"pmcid", "pmc"}:
            value = node_text(el)
            return value if value.upper().startswith("PMC") else f"PMC{value}"
    return None


def _pub_year(soup: BeautifulSoup) -> int | None:
    for date in tags(soup.select("article-meta pub-date")):
        year = date.find("year")
        if isinstance(year, Tag) and node_text(year).isdigit():
            return int(node_text(year))
    return None


def _doc_type(article_type: str, title: str, abstract: str) -> str:
    base = ARTICLE_TYPE_TO_DOC_TYPE.get(article_type, "other")
    cues = f"{title} {abstract[:1500]}"
    if base in {"other", "review"} and _SR_CUE.search(cues):
        return "systematic_review"
    if base == "other" and _RCT_CUE.search(cues):
        return "rct"
    return base


def _sec_blocks(node: Tag) -> list[Block]:
    blocks: list[Block] = []
    for child in tags(node.children):
        if child.name == "p":
            blocks.append(Block("paragraph", node_text(child)))
        elif child.name == "list":
            for item in tags(child.find_all("list-item", recursive=False)):
                blocks.append(Block("list_item", node_text(item)))
        elif child.name == "table-wrap":
            label = child.find("label")
            caption = child.find("caption")
            cap = " ".join(node_text(x) for x in (label, caption) if isinstance(x, Tag))
            if cap:
                blocks.append(Block("caption", cap))
            table = child.find("table")
            if isinstance(table, Tag):
                blocks.extend(b for b in table_blocks(table) if b.kind != "caption")
        elif child.name == "fig":
            caption = child.find("caption")
            if isinstance(caption, Tag):
                blocks.append(Block("caption", node_text(caption)))
        elif child.name in {"disp-quote", "boxed-text"}:
            blocks.extend(_sec_blocks(child))
    return [b for b in blocks if b.text]


def _walk(sec: Tag, path: list[str], out: list[ParsedSection]) -> None:
    title = sec.find("title", recursive=False)
    heading = node_text(title) if isinstance(title, Tag) else ""
    here = [*path, heading] if heading else path
    blocks = _sec_blocks(sec)
    if blocks:
        out.append(
            ParsedSection(
                heading_path=here,
                anchor=str(sec.get("id")) if sec.get("id") else None,
                blocks=blocks,
            )
        )
    for child in tags(sec.find_all("sec", recursive=False)):
        _walk(child, here, out)


def parse_jats(xml: str, *, require_open_license: bool = True) -> ParsedDocument:
    soup = BeautifulSoup(xml, "lxml-xml")
    article = soup.find("article")
    if not isinstance(article, Tag):
        raise ParseError("JATS <article> not found")
    license_kind, license_ref = _license(soup)
    if require_open_license and license_kind is None:
        raise LicenseRejected(f"license not CC BY/CC0: {license_ref or 'missing'}")

    for el in tags(soup.find_all(["ref-list", "back", "fn-group", "ack"])):
        el.decompose()
    for xref in tags(soup.find_all("xref", attrs={"ref-type": "bibr"})):
        xref.decompose()
    prepare_tree(article)

    title_el = soup.find("article-title")
    title = node_text(title_el) if isinstance(title_el, Tag) else ""
    sections: list[ParsedSection] = []
    abstract = soup.find("abstract")
    abstract_text = ""
    if isinstance(abstract, Tag):
        abstract_text = node_text(abstract)
        blocks = _sec_blocks(abstract)
        if blocks:
            sections.append(ParsedSection(heading_path=["Abstract"], blocks=blocks))
        for sec in tags(abstract.find_all("sec", recursive=False)):
            _walk(sec, ["Abstract"], sections)
    body = soup.find("body")
    if isinstance(body, Tag):
        loose = _sec_blocks(body)
        if loose:
            sections.append(ParsedSection(heading_path=[], blocks=loose))
        for sec in tags(body.find_all("sec", recursive=False)):
            _walk(sec, [], sections)
    if not sections:
        raise ParseError("no JATS content extracted")
    return ParsedDocument(
        sections=sections,
        title=title or None,
        authors=_authors(soup),
        license=f"{(license_kind or 'unknown').upper()} ({license_ref})"
        if license_ref
        else license_kind,
        pub_year=_pub_year(soup),
        doc_type=_doc_type(str(article.get("article-type", "")), title, abstract_text),
    )
