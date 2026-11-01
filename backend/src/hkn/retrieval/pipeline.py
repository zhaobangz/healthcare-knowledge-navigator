"""Retrieval orchestration: query variants × (dense ∥ FTS) arms → RRF → rerank → top-k.

`RetrievalConfig` is the ablation ladder: every rung of the retrieval eval is a config
here, so the eval measures exactly the code path the API serves.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hkn.config import Settings, get_settings
from hkn.ingest.embed import embed_passage_query, embed_query
from hkn.models import Chunk, Document, Section
from hkn.retrieval.fusion import rrf_scores
from hkn.retrieval.plan_query import QueryPlan
from hkn.retrieval.search import (
    Filters,
    Hit,
    dense_search,
    fts_search,
    informative_terms,
    question_terms,
)


@dataclass(frozen=True)
class RetrievalConfig:
    dense: bool = True
    fts: bool = True
    use_query_plan: bool = True
    use_hyde: bool = True
    use_sub_questions: bool = True
    rerank: bool = True
    context_mode: str = "heading"  # plain | heading | llm
    top_k: int = 8
    dense_candidates: int = 50
    fts_candidates: int = 50
    rerank_candidates: int = 30
    rrf_k: int = 60
    ef_search: int = 100
    soft_filter_min_hits: int = 5
    # Small-to-big sends whole sections, so a third chunk from the same section adds no
    # new text to the prompt; capping per section lets top-k reach neighboring sections.
    max_chunks_per_section: int = 2
    # Drop corpus-frequent words from the OR'd full-text query (ts_rank_cd has no IDF).
    idf_prune: bool = True

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalConfig:
        s = settings or get_settings()
        return cls(
            use_query_plan=s.enable_query_plan,
            use_hyde=s.enable_hyde,
            use_sub_questions=s.enable_sub_questions,
            rerank=s.enable_rerank,
            context_mode="heading" if s.use_context_prefix else "plain",
            top_k=s.top_k,
            dense_candidates=s.dense_candidates,
            fts_candidates=s.fts_candidates,
            rerank_candidates=s.rerank_candidates,
            rrf_k=s.rrf_k,
            ef_search=s.hnsw_ef_search,
            soft_filter_min_hits=s.soft_filter_min_hits,
        )

    def with_(self, **changes: Any) -> RetrievalConfig:
        return replace(self, **changes)


@dataclass
class Candidate:
    chunk_id: int
    section_id: int
    document_id: int
    first_sentence_ord: int
    last_sentence_ord: int
    text: str
    context_prefix: str
    source_key: str
    doc_title: str
    org: str
    doc_type: str
    pub_year: int | None
    heading_path: list[str]
    arm_ranks: dict[str, int] = field(default_factory=dict)
    arm_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float | None = None

    def rerank_text(self) -> str:
        return f"{self.context_prefix}\n{self.text}" if self.context_prefix else self.text


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    config: RetrievalConfig
    hard_filters: Filters
    soft_filters_applied: Filters | None
    soft_filters_dropped: bool
    queries: dict[str, str]
    keywords: list[str]
    timings_ms: dict[str, float]
    pool_size: int

    def trace(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "queries": self.queries,
            "keywords": self.keywords,
            "hard_filters": asdict(self.hard_filters),
            "soft_filters": asdict(self.soft_filters_applied)
            if self.soft_filters_applied
            else None,
            "soft_filters_dropped": self.soft_filters_dropped,
            "pool_size": self.pool_size,
            "timings_ms": self.timings_ms,
            "candidates": [
                {
                    "chunk_id": c.chunk_id,
                    "doc": c.doc_title
                    if c.doc_title.startswith(c.org)
                    else f"{c.org} — {c.doc_title}",
                    "heading": " › ".join(c.heading_path),
                    "rrf": round(c.rrf_score, 5),
                    "rerank": None if c.rerank_score is None else round(c.rerank_score, 3),
                    "arm_ranks": c.arm_ranks,
                    "preview": c.text[:160],
                }
                for c in self.candidates
            ],
        }


def _plan_filters(plan: QueryPlan | None) -> Filters | None:
    if plan is None:
        return None
    sf = plan.soft_filters
    soft = Filters(
        doc_types=list(sf.doc_types) or None,
        specialties=list(sf.specialties) or None,
        year_min=sf.year_min,
    )
    return None if soft.is_empty() else soft


def _lexical_terms(session: Session, question: str, config: RetrievalConfig) -> list[str]:
    terms = question_terms(question)
    return (
        informative_terms(session, terms, mode=config.context_mode) if config.idf_prune else terms
    )


def _run_arms(
    session: Session,
    question: str,
    plan: QueryPlan | None,
    filters: Filters,
    config: RetrievalConfig,
    timings: dict[str, float],
) -> tuple[dict[str, list[Hit]], dict[str, str], list[str]]:
    queries: dict[str, str] = {"original": question}
    if plan is not None and config.use_query_plan:
        queries["rewritten"] = plan.rewritten_query
        if config.use_sub_questions:
            for i, sq in enumerate(plan.sub_questions[:3]):
                queries[f"sub{i}"] = sq
    keywords = (
        plan.keywords
        if plan is not None and config.use_query_plan
        else _lexical_terms(session, question, config)
    )

    arms: dict[str, list[Hit]] = {}
    t0 = time.perf_counter()
    if config.dense:
        for name, q in queries.items():
            arms[f"dense:{name}"] = dense_search(
                session, embed_query(q), filters=filters, limit=config.dense_candidates,
                mode=config.context_mode, ef_search=config.ef_search,
            )  # fmt: skip
        if plan is not None and config.use_hyde and plan.hyde_passage:
            arms["dense:hyde"] = dense_search(
                session, embed_passage_query(plan.hyde_passage), filters=filters,
                limit=config.dense_candidates, mode=config.context_mode, ef_search=config.ef_search,
            )  # fmt: skip
    timings["dense"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    if config.fts:
        arms["fts:keywords"] = fts_search(
            session,
            keywords,
            filters=filters,
            limit=config.fts_candidates,
            mode=config.context_mode,
        )
        if plan is not None and config.use_query_plan:
            # The user's own words are a separate lexical signal from the planner's keywords.
            arms["fts:original"] = fts_search(
                session,
                _lexical_terms(session, question, config),
                filters=filters,
                limit=config.fts_candidates, mode=config.context_mode,
            )  # fmt: skip
    timings["fts"] = (time.perf_counter() - t0) * 1000
    return arms, queries, keywords


def _load_candidates(session: Session, chunk_ids: list[int]) -> dict[int, Candidate]:
    rows = session.execute(
        select(Chunk, Section.heading_path, Document)
        .join(Section, Section.id == Chunk.section_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    out: dict[int, Candidate] = {}
    for chunk, heading_path, doc in rows:
        out[chunk.id] = Candidate(
            chunk_id=chunk.id,
            section_id=chunk.section_id,
            document_id=chunk.document_id,
            first_sentence_ord=chunk.first_sentence_ord,
            last_sentence_ord=chunk.last_sentence_ord,
            text=chunk.text,
            context_prefix=chunk.context_prefix,
            source_key=doc.source_key,
            doc_title=doc.title,
            org=doc.org,
            doc_type=doc.doc_type,
            pub_year=doc.pub_year,
            heading_path=list(heading_path or []),
        )
    return out


def _cap_per_section(candidates: list[Candidate], cap: int) -> list[Candidate]:
    if cap <= 0:
        return candidates
    seen: dict[int, int] = {}
    out = []
    for cand in candidates:
        seen[cand.section_id] = seen.get(cand.section_id, 0) + 1
        if seen[cand.section_id] <= cap:
            out.append(cand)
    return out


def retrieve(
    session: Session,
    question: str,
    *,
    plan: QueryPlan | None = None,
    hard_filters: Filters | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    from hkn.retrieval.rerank import rerank_scores  # lazy: loads a model

    config = config or RetrievalConfig.from_settings()
    hard = hard_filters or Filters()
    timings: dict[str, float] = {}

    soft = _plan_filters(plan) if config.use_query_plan else None
    soft_dropped = False
    effective = hard.merged(soft) if soft else hard
    arms, queries, keywords = _run_arms(session, question, plan, effective, config, timings)
    if soft is not None:
        pooled = {h.chunk_id for hits in arms.values() for h in hits}
        if len(pooled) < config.soft_filter_min_hits:
            # Inferred filters starved retrieval: fall back to the user's filters only.
            soft_dropped = True
            arms, queries, keywords = _run_arms(session, question, plan, hard, config, timings)

    rankings = {name: [h.chunk_id for h in hits] for name, hits in arms.items() if hits}
    fused = rrf_scores(list(rankings.values()), k=config.rrf_k)
    order = sorted(fused, key=lambda cid: -fused[cid])
    pool = order[: max(config.rerank_candidates, config.top_k)]
    loaded = _load_candidates(session, pool)
    candidates = [loaded[cid] for cid in pool if cid in loaded]
    for cand in candidates:
        cand.rrf_score = fused[cand.chunk_id]
        for name, hits in arms.items():
            for rank, hit in enumerate(hits, start=1):
                if hit.chunk_id == cand.chunk_id:
                    cand.arm_ranks[name] = rank
                    cand.arm_scores[name] = round(hit.score, 5)
                    break

    t0 = time.perf_counter()
    if config.rerank and candidates:
        scores = rerank_scores(question, [c.rerank_text() for c in candidates])
        for cand, score in zip(candidates, scores, strict=True):
            cand.rerank_score = score
        candidates.sort(key=lambda c: -(c.rerank_score or 0.0))
    final = _cap_per_section(candidates, config.max_chunks_per_section)[: config.top_k]
    if not config.rerank and final:
        # Ordering stays RRF (that is the ablation), but the returned chunks are still
        # scored: the gate and the confidence features need a relevance signal either way.
        for cand, score in zip(
            final, rerank_scores(question, [c.rerank_text() for c in final]), strict=True
        ):
            cand.rerank_score = score
    timings["rerank"] = (time.perf_counter() - t0) * 1000

    return RetrievalResult(
        candidates=final,
        config=config,
        hard_filters=hard,
        soft_filters_applied=None if soft_dropped else soft,
        soft_filters_dropped=soft_dropped,
        queries=queries,
        keywords=keywords,
        timings_ms={k: round(v, 1) for k, v in timings.items()},
        pool_size=len(fused),
    )
