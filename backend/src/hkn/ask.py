"""End-to-end question answering as one ordered event stream.

    trace → sources → confidence_preliminary → [notice] →
        status · (block_start · text* · citation* · block_stop)* · [served_by]
    → confidence → done
  or, on any gate, refusal or error:  … → insufficient | refusal | error → done

The same steps back the CLI, the SSE API and the evals (`prepare` and `finalize` are
shared; only the answer stream is sync or async), so an eval number always describes
the code path users get.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from hkn.confidence.evidence import CitedDoc, evidence_strength
from hkn.confidence.scorer import (
    Reliability,
    compute_features,
    preliminary_retrieval_strength,
    score_reliability,
)
from hkn.config import Settings, get_settings
from hkn.db import session_scope
from hkn.generation.answer import AnswerEvent, CollectedAnswer, astream_answer, stream_answer
from hkn.generation.extractive import EXTRACTIVE_NOTICE, extractive_events
from hkn.retrieval.expand import SourceSection, expand_to_sections
from hkn.retrieval.gates import GateResult, gate_answer, gate_retrieval, retrieval_threshold
from hkn.retrieval.pipeline import RetrievalConfig, RetrievalResult, retrieve
from hkn.retrieval.plan_query import QueryPlan, plan_query
from hkn.retrieval.search import Filters

Event = dict[str, Any]  # {"event": name, "data": {...}}

PERSONAL_ADVICE_NOTICE = (
    "This tool summarizes published guidance for health professionals. It cannot advise on "
    "an individual's care; discuss personal health decisions with a clinician."
)


@dataclass
class Prepared:
    question: str
    plan: QueryPlan | None
    plan_error: str | None
    retrieval: RetrievalResult | None
    sources: list[SourceSection]
    gate: GateResult
    stage: str | None = None  # which gate failed: scope | retrieval

    @property
    def top_rerank(self) -> float | None:
        if self.retrieval is None:
            return None
        return max(
            (c.rerank_score for c in self.retrieval.candidates if c.rerank_score is not None),
            default=None,
        )


@dataclass
class AskResult:
    question: str
    status: str  # answered | insufficient | refused | error
    answer: CollectedAnswer
    sources: list[SourceSection]
    reliability: Reliability | None
    evidence: dict[str, Any] | None
    reason: str | None
    trace: dict[str, Any]
    events: list[Event] = field(default_factory=list)


def _ev(name: str, data: dict[str, Any]) -> Event:
    return {"event": name, "data": data}


def prepare(
    question: str,
    *,
    filters: Filters | None = None,
    config: RetrievalConfig | None = None,
    plan: QueryPlan | None = None,
    settings: Settings | None = None,
) -> Prepared:
    settings = settings or get_settings()
    config = config or RetrievalConfig.from_settings(settings)
    plan_error = None
    if plan is None and config.use_query_plan:
        plan, plan_error = plan_query(question, settings=settings)
    if plan is not None and not plan.is_clinical:
        return Prepared(
            question, plan, plan_error, None, [],
            GateResult(False, "This tool answers clinical questions from medical guidelines and literature."),
            stage="scope",
        )  # fmt: skip
    with session_scope() as session:
        retrieval = retrieve(session, question, plan=plan, hard_filters=filters, config=config)
        gate = gate_retrieval(
            retrieval.candidates,
            min_rerank_logit=retrieval_threshold(settings),
            rerank_enabled=True,  # candidates are always rerank-scored (see retrieve)
        )
        sources = (
            expand_to_sections(
                session, retrieval.candidates, window_tokens=settings.section_window_tokens
            )
            if gate.passed
            else []
        )
    return Prepared(
        question, plan, plan_error, retrieval, sources, gate, None if gate.passed else "retrieval"
    )


def _trace(p: Prepared) -> dict[str, Any]:
    return {
        "plan": None if p.plan is None else p.plan.model_dump(),
        "plan_error": p.plan_error,
        "retrieval": None if p.retrieval is None else p.retrieval.trace(),
    }


def sources_payload(sources: list[SourceSection]) -> list[dict[str, Any]]:
    return [
        {
            "index": s.index,
            "number": s.index + 1,
            "title": s.title,
            "doc_title": s.doc_title,
            "org": s.org,
            "doc_type": s.doc_type,
            "pub_year": s.pub_year,
            "url": s.source_url,
            "license": s.license,
            "authors": s.authors,
            "superseded": s.superseded,
            "heading_path": s.heading_path,
            "rerank": None if s.best_rerank is None else round(s.best_rerank, 3),
            "sentences": [
                {"id": x.id, "text": x.text, "kind": x.kind, "matched": x.matched}
                for x in s.sentences
            ],
        }
        for s in sources
    ]


def _extractive() -> bool:
    return get_settings().answer_mode == "extractive"


def _answer_stream(question: str, sources: list[SourceSection]) -> Iterator[AnswerEvent]:
    if _extractive():
        return extractive_events(question, sources)
    return stream_answer(question, sources)


def _opening_events(p: Prepared) -> list[Event]:
    out = [_ev("trace", _trace(p)), _ev("sources", {"sources": sources_payload(p.sources)})]
    out.append(_ev("confidence_preliminary", preliminary_retrieval_strength(p.top_rerank)))
    if _extractive() and p.gate.passed:
        out.append(_ev("notice", {"kind": "extractive", "text": EXTRACTIVE_NOTICE}))
    if p.plan is not None and p.plan.asks_personal_advice:
        out.append(_ev("notice", {"kind": "personal_advice", "text": PERSONAL_ADVICE_NOTICE}))
    return out


def finalize(
    p: Prepared,
    answer: CollectedAnswer,
    *,
    nli_fn: Any | None = None,
    relevance_fn: Any | None = None,
) -> tuple[str, list[Event], Reliability | None, dict | None, str | None]:
    """Gate 2 + confidence. Returns (status, closing events, reliability, evidence, reason)."""
    if answer.error is not None:
        return (
            "error",
            [_ev("error", answer.error), _ev("done", {"status": "error"})],
            None,
            None,
            str(answer.error.get("message") or "generation error"),
        )
    if answer.refused:
        data = {"category": answer.refusal_category}
        return (
            "refused",
            [_ev("refusal", data), _ev("done", {"status": "refused"})],
            None,
            None,
            "declined",
        )
    gate = gate_answer(answer)
    if not gate.passed:
        return (
            "insufficient",
            [_ev("insufficient", {"stage": "answer", "reason": gate.reason}), _ev("done", {"status": "insufficient"})],
            None, None, gate.reason,
        )  # fmt: skip

    features = compute_features(
        answer,
        p.sources,
        top_rerank_logit=p.top_rerank,
        question=p.question,
        nli_fn=nli_fn,
        relevance_fn=relevance_fn,
    )
    reliability = score_reliability(features)
    by_index = {s.index: s for s in p.sources}
    cited_docs = [
        CitedDoc(s.document_id, s.doc_type, s.org, s.doc_title, s.pub_year, s.superseded)
        for s in (by_index[c.source_index] for c in answer.citations if c.source_index in by_index)
    ]
    evidence = asdict(evidence_strength(cited_docs, current_year=dt.date.today().year))
    confidence = {
        "reliability": reliability.to_dict(),
        "evidence": evidence,
        "claims": [{**asdict(c), "status": c.status} for c in features.claims],
    }
    return (
        "answered",
        [_ev("confidence", confidence), _ev("done", {"status": "answered", **(answer.done or {})})],
        reliability,
        evidence,
        None,
    )


def _gate_events(p: Prepared) -> list[Event]:
    return [
        _ev("insufficient", {"stage": p.stage, "reason": p.gate.reason}),
        _ev("done", {"status": "insufficient"}),
    ]


def _answer_event(e: AnswerEvent) -> Event:
    return _ev(e.type, e.data)


def ask_events(question: str, **kwargs: Any) -> Iterator[Event]:
    """Synchronous event stream (CLI, evals)."""
    p = prepare(question, **kwargs)
    yield from _opening_events(p)
    if not p.gate.passed:
        yield from _gate_events(p)
        return
    answer = CollectedAnswer()
    for e in _answer_stream(question, p.sources):
        answer.apply(e)
        if e.type not in {"done", "refusal", "error"}:
            yield _answer_event(e)
    _, closing, _, _, _ = finalize(p, answer)
    yield from closing


async def _replay(events: list[AnswerEvent]) -> AsyncIterator[AnswerEvent]:
    for e in events:
        yield e


async def ask_events_async(question: str, **kwargs: Any) -> AsyncIterator[Event]:
    """Async event stream for the SSE API. CPU-bound steps (embedding, reranking, NLI)
    run in worker threads so the event loop keeps serving other streams."""
    p = await asyncio.to_thread(lambda: prepare(question, **kwargs))
    for ev in _opening_events(p):
        yield ev
    if not p.gate.passed:
        for ev in _gate_events(p):
            yield ev
        return
    answer = CollectedAnswer()
    if _extractive():
        stream = _replay(
            await asyncio.to_thread(lambda: list(extractive_events(question, p.sources)))
        )
    else:
        stream = astream_answer(question, p.sources)
    async for e in stream:
        answer.apply(e)
        if e.type not in {"done", "refusal", "error"}:
            yield _answer_event(e)
    _, closing, _, _, _ = await asyncio.to_thread(finalize, p, answer)
    for ev in closing:
        yield ev


def ask(question: str, **kwargs: Any) -> AskResult:
    """Run the full pipeline and return the collected result (CLI, evals)."""
    p = prepare(question, **kwargs)
    events = _opening_events(p)
    if not p.gate.passed:
        events += _gate_events(p)
        return AskResult(
            question,
            "insufficient",
            CollectedAnswer(),
            p.sources,
            None,
            None,
            p.gate.reason,
            _trace(p),
            events,
        )
    answer = CollectedAnswer()
    for e in _answer_stream(question, p.sources):
        answer.apply(e)
        if e.type not in {"done", "refusal", "error"}:  # `finalize` emits the terminal events
            events.append(_answer_event(e))
    status, closing, reliability, evidence, reason = finalize(p, answer)
    events += closing
    return AskResult(
        question, status, answer, p.sources, reliability, evidence, reason, _trace(p), events
    )
