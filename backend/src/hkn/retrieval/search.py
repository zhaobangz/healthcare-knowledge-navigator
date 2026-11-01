"""Dense and full-text search arms over chunks, with metadata filtering (concepts 5, 7, 8).

Filters are applied in SQL before ranking. Filtered approximate-nearest-neighbour search
is the classic way to silently get too few rows (HNSW visits `ef_search` candidates and
then filters them away), so the dense arm enables pgvector's iterative index scan and a
larger `ef_search`.

The FTS arm ORs terms instead of using `websearch_to_tsquery`, which ANDs every word: a
long natural-language question would otherwise match nothing. `ts_rank_cd` still ranks
chunks that match more of the terms (and denser) higher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from hkn.models import CONTEXT_MODES


@dataclass
class Filters:
    doc_types: list[str] | None = None
    orgs: list[str] | None = None
    specialties: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    exclude_superseded: bool = False
    # Used by the eval's "gold document withheld" degraded configuration.
    exclude_document_ids: list[int] | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.doc_types,
                self.orgs,
                self.specialties,
                self.year_min,
                self.year_max,
                self.exclude_superseded,
                self.exclude_document_ids,
            ]
        )

    def merged(self, other: Filters) -> Filters:
        """`self` wins where both set a field (hard UI filters override soft ones)."""
        return Filters(
            doc_types=self.doc_types or other.doc_types,
            orgs=self.orgs or other.orgs,
            specialties=self.specialties or other.specialties,
            year_min=self.year_min if self.year_min is not None else other.year_min,
            year_max=self.year_max if self.year_max is not None else other.year_max,
            exclude_superseded=self.exclude_superseded or other.exclude_superseded,
            exclude_document_ids=self.exclude_document_ids or other.exclude_document_ids,
        )


@dataclass
class Hit:
    chunk_id: int
    score: float


@dataclass
class ArmResult:
    name: str
    hits: list[Hit] = field(default_factory=list)


def _where(filters: Filters, params: dict[str, Any]) -> str:
    clauses: list[str] = []
    if filters.doc_types:
        clauses.append("d.doc_type = ANY(:f_doc_types)")
        params["f_doc_types"] = list(filters.doc_types)
    if filters.orgs:
        clauses.append("d.org = ANY(:f_orgs)")
        params["f_orgs"] = list(filters.orgs)
    if filters.specialties:
        clauses.append("d.specialty && CAST(:f_specialties AS varchar[])")
        params["f_specialties"] = list(filters.specialties)
    if filters.year_min is not None:
        clauses.append("d.pub_year >= :f_year_min")
        params["f_year_min"] = filters.year_min
    if filters.year_max is not None:
        clauses.append("d.pub_year <= :f_year_max")
        params["f_year_max"] = filters.year_max
    if filters.exclude_superseded:
        clauses.append("NOT d.superseded")
    if filters.exclude_document_ids:
        clauses.append("NOT (d.id = ANY(:f_exclude_docs))")
        params["f_exclude_docs"] = list(filters.exclude_document_ids)
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _check_mode(mode: str) -> str:
    if mode not in CONTEXT_MODES:  # interpolated into SQL below — whitelist only
        raise ValueError(f"unknown context mode {mode!r}")
    return mode


def dense_search(
    session: Session,
    vector: list[float],
    *,
    filters: Filters,
    limit: int,
    mode: str = "heading",
    ef_search: int = 100,
) -> list[Hit]:
    col = f"c.embedding_{_check_mode(mode)}"
    params: dict[str, Any] = {"q": str(vector), "limit": limit}
    where = _where(filters, params)
    session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = session.execute(
        text(
            f"""
            SELECT c.id, 1 - ({col} <=> CAST(:q AS vector)) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {col} IS NOT NULL {where}
            ORDER BY {col} <=> CAST(:q AS vector)
            LIMIT :limit
            """
        ),
        params,
    ).all()
    # relaxed_order may return results slightly out of order; restore exact order.
    hits = [Hit(int(r[0]), float(r[1])) for r in rows]
    return sorted(hits, key=lambda h: -h.score)


_TERM_CLEAN = re.compile(r"[^\w\s\-/.%≥≤]", re.UNICODE)


def or_terms(terms: list[str]) -> list[str]:
    """Normalize keyword phrases for `plainto_tsquery` (which ANDs within a phrase)."""
    out: list[str] = []
    for term in terms:
        cleaned = _TERM_CLEAN.sub(" ", term).strip()
        if cleaned and cleaned.lower() not in {t.lower() for t in out}:
            out.append(cleaned)
    return out


def question_terms(question: str) -> list[str]:
    """Fallback keywords when no query plan is available: every word, OR'd."""
    return or_terms(question.split())


_DF_CACHE: dict[str, tuple[int, dict[str, int]]] = {}


def _document_frequencies(session: Session, mode: str) -> tuple[int, dict[str, int]]:
    """Lexeme → number of chunks containing it, cached per process and context mode.
    Invalidated when the chunk count changes (i.e. after an ingest)."""
    col = f"tsv_{_check_mode(mode)}"
    n = int(session.execute(text("SELECT count(*) FROM chunks")).scalar() or 0)
    cached = _DF_CACHE.get(mode)
    if cached is None or cached[0] != n:
        rows = session.execute(
            text(f"SELECT word, ndoc FROM ts_stat('SELECT {col} FROM chunks')")
        ).all()
        _DF_CACHE[mode] = (n, {str(w): int(d) for w, d in rows})
    return _DF_CACHE[mode]


def informative_terms(
    session: Session,
    terms: list[str],
    *,
    mode: str = "heading",
    max_df: float = 0.05,
    keep_min: int = 3,
) -> list[str]:
    """IDF pruning for the OR'd query: Postgres `ts_rank_cd` has no inverse document
    frequency, so frequent words ("recommended", "treatment") would dominate the ranking.
    Keep terms whose rarest lexeme occurs in at most `max_df` of chunks, and always the
    `keep_min` rarest terms so a question never loses its lexical arm entirely."""
    n, df = _document_frequencies(session, mode)
    if not terms or n == 0:
        return terms
    rows = session.execute(
        text(
            "SELECT t, lexeme FROM unnest(CAST(:terms AS text[])) AS t, "
            "unnest(tsvector_to_array(to_tsvector('english', t))) AS lexeme"
        ),
        {"terms": terms},
    ).all()
    rarity: dict[str, int] = {}
    for term, lexeme in rows:
        rarity[term] = min(rarity.get(term, n), df.get(lexeme, 0))
    ranked = sorted((t for t in terms if t in rarity), key=lambda t: rarity[t])
    kept = [t for t in ranked if rarity[t] <= max_df * n]
    return kept if len(kept) >= keep_min else ranked[:keep_min]


def fts_search(
    session: Session,
    terms: list[str],
    *,
    filters: Filters,
    limit: int,
    mode: str = "heading",
) -> list[Hit]:
    terms = or_terms(terms)
    if not terms:
        return []
    col = f"c.tsv_{_check_mode(mode)}"
    params: dict[str, Any] = {"limit": limit}
    parts = []
    for i, term in enumerate(terms[:40]):
        params[f"t{i}"] = term
        parts.append(f"plainto_tsquery('english', :t{i})")
    where = _where(filters, params)
    query = " || ".join(parts)
    rows = session.execute(
        text(
            f"""
            WITH q AS (SELECT ({query}) AS tsq)
            SELECT c.id, ts_rank_cd({col}, q.tsq, 32) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id, q
            WHERE numnode(q.tsq) > 0 AND {col} @@ q.tsq {where}
            ORDER BY score DESC, c.id
            LIMIT :limit
            """
        ),
        params,
    ).all()
    return [Hit(int(r[0]), float(r[1])) for r in rows]
