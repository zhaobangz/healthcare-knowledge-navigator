"""Small-to-big expansion (concept 12).

Retrieval matched small chunks; the answer model gets the surrounding section instead,
because a guideline regimen is often split across a table, its footnotes and the
paragraph that qualifies it. Each matched section is widened symmetrically around the
matched sentences up to `section_window_tokens`, and every sentence becomes its own
search-result text block — Claude cites whole blocks, so sentence blocks give
sentence-level citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hkn.ingest.chunk import approx_tokens
from hkn.models import Document, Section, Sentence
from hkn.retrieval.pipeline import Candidate


@dataclass
class SourceSentence:
    id: int
    ord: int
    text: str
    kind: str
    char_start: int
    char_end: int
    matched: bool


@dataclass
class SourceSection:
    """One search_result block in the answer request. `index` is its position in the
    request, which is exactly what citations' `search_result_index` refers to."""

    index: int
    section_id: int
    document_id: int
    source_key: str
    doc_title: str
    org: str
    doc_type: str
    pub_year: int | None
    url: str
    license: str
    authors: str
    superseded: bool
    heading_path: list[str]
    anchor: str | None
    sentences: list[SourceSentence] = field(default_factory=list)
    best_rerank: float | None = None
    best_rank: int = 0
    chunk_ids: list[int] = field(default_factory=list)

    @property
    def title(self) -> str:
        year = (
            f" ({self.pub_year})"
            if self.pub_year and str(self.pub_year) not in self.doc_title
            else ""
        )
        path = " › ".join(self.heading_path)
        return f"{self.org} — {self.doc_title}{year}" + (f" › {path}" if path else "")

    @property
    def source_url(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url


def _window(sentences: list[Sentence], matched: set[int], budget: int) -> list[Sentence]:
    """Grow a contiguous window outward from the matched span, one sentence per side per
    step, stopping each side when the next sentence would exceed the budget. The matched
    span itself is always kept, even if it alone exceeds the budget."""
    by_ord = {s.ord: s for s in sentences}
    if not by_ord or not matched:
        return []
    lo, hi = min(matched), max(matched)
    chosen = {o for o in range(lo, hi + 1) if o in by_ord}
    used = sum(approx_tokens(by_ord[o].text) for o in chosen)
    left, right = lo - 1, hi + 1
    left_open, right_open = True, True
    while left_open or right_open:
        for side in ("left", "right"):
            o = left if side == "left" else right
            is_open = left_open if side == "left" else right_open
            if not is_open:
                continue
            if o not in by_ord or used + approx_tokens(by_ord[o].text) > budget:
                if side == "left":
                    left_open = False
                else:
                    right_open = False
                continue
            chosen.add(o)
            used += approx_tokens(by_ord[o].text)
            if side == "left":
                left -= 1
            else:
                right += 1
    return [by_ord[o] for o in sorted(chosen)]


def expand_to_sections(
    session: Session, candidates: list[Candidate], *, window_tokens: int, max_sections: int = 8
) -> list[SourceSection]:
    """Group ranked chunks by section (best rank first) and build windowed sources."""
    order: list[int] = []
    matched: dict[int, set[int]] = {}
    best: dict[int, Candidate] = {}
    chunk_ids: dict[int, list[int]] = {}
    for cand in candidates:
        if cand.section_id not in matched:
            order.append(cand.section_id)
            matched[cand.section_id] = set()
            best[cand.section_id] = cand
            chunk_ids[cand.section_id] = []
        matched[cand.section_id].update(range(cand.first_sentence_ord, cand.last_sentence_ord + 1))
        chunk_ids[cand.section_id].append(cand.chunk_id)
    order = order[:max_sections]
    if not order:
        return []

    sections = {
        s.id: (s, d)
        for s, d in session.execute(
            select(Section, Document)
            .join(Document, Document.id == Section.document_id)
            .where(Section.id.in_(order))
        ).all()
    }
    sentence_rows = session.scalars(
        select(Sentence)
        .where(Sentence.section_id.in_(order))
        .order_by(Sentence.section_id, Sentence.ord)
    ).all()
    by_section: dict[int, list[Sentence]] = {}
    for row in sentence_rows:
        by_section.setdefault(row.section_id, []).append(row)

    out: list[SourceSection] = []
    for rank, section_id in enumerate(order, start=1):
        if section_id not in sections:
            continue
        section, doc = sections[section_id]
        window = _window(by_section.get(section_id, []), matched[section_id], window_tokens)
        out.append(
            SourceSection(
                index=len(out),
                section_id=section_id,
                document_id=doc.id,
                source_key=doc.source_key,
                doc_title=doc.title,
                org=doc.org,
                doc_type=doc.doc_type,
                pub_year=doc.pub_year,
                url=doc.url,
                license=doc.license,
                authors=doc.authors,
                superseded=doc.superseded,
                heading_path=list(section.heading_path or []),
                anchor=section.anchor,
                sentences=[
                    SourceSentence(
                        id=s.id,
                        ord=s.ord,
                        text=s.text,
                        kind=s.kind,
                        char_start=s.char_start,
                        char_end=s.char_end,
                        matched=s.ord in matched[section_id],
                    )
                    for s in window
                ],
                best_rerank=best[section_id].rerank_score,
                best_rank=rank,
                chunk_ids=chunk_ids[section_id],
            )
        )
    return out
