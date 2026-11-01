"""FastAPI app: SSE question answering plus corpus browsing endpoints.

uvicorn hkn.api.main:app --port 8002
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, text

from hkn.api.schemas import AskBody, DocumentOut, FacetsOut, SectionOut, SentenceOut
from hkn.api.sse import SSE_HEADERS, relay
from hkn.ask import ask_events_async
from hkn.config import get_settings
from hkn.db import session_scope
from hkn.models import Chunk, Document, Section, Sentence

log = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Healthcare Knowledge Navigator",
    description="Cited, confidence-scored answers from clinical guidelines and literature.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
            chunks = s.scalar(select(func.count()).select_from(Chunk)) or 0
    except Exception as exc:  # report, don't crash: the UI shows a banner
        log.warning("health check failed: %s", exc)
        raise HTTPException(503, "database unavailable") from exc
    return {"status": "ok", "chunks": chunks}


@app.post("/ask")
async def ask_endpoint(body: AskBody, request: Request) -> StreamingResponse:
    """Stream an answer as Server-Sent Events. Questions are not logged."""
    events = ask_events_async(body.question.strip(), filters=body.filters.to_filters())
    return StreamingResponse(
        relay(request, events), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.get("/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    with session_scope() as s:
        sec_counts = dict(
            s.execute(select(Section.document_id, func.count()).group_by(Section.document_id)).all()
        )
        chunk_counts = dict(
            s.execute(select(Chunk.document_id, func.count()).group_by(Chunk.document_id)).all()
        )
        docs = s.scalars(select(Document).order_by(Document.org, Document.title)).all()
        return [
            DocumentOut(
                id=d.id,
                source_key=d.source_key,
                title=d.title,
                org=d.org,
                doc_type=d.doc_type,
                specialty=list(d.specialty or []),
                pub_year=d.pub_year,
                url=d.url,
                license=d.license,
                authors=d.authors,
                superseded=d.superseded,
                sections=sec_counts.get(d.id, 0),
                chunks=chunk_counts.get(d.id, 0),
            )
            for d in docs
        ]


@app.get("/facets", response_model=FacetsOut)
def facets() -> FacetsOut:
    with session_scope() as s:
        docs = s.scalars(select(Document)).all()
        n_chunks = s.scalar(select(func.count()).select_from(Chunk)) or 0
    years = [d.pub_year for d in docs if d.pub_year]
    return FacetsOut(
        doc_types=sorted({d.doc_type for d in docs}),
        orgs=sorted({d.org for d in docs}),
        specialties=sorted({sp for d in docs for sp in (d.specialty or [])}),
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
        documents=len(docs),
        chunks=n_chunks,
    )


@app.get("/sections/{section_id}", response_model=SectionOut)
def section(section_id: int) -> SectionOut:
    with session_scope() as s:
        sec = s.get(Section, section_id)
        if sec is None:
            raise HTTPException(404, "section not found")
        rows = s.scalars(
            select(Sentence).where(Sentence.section_id == section_id).order_by(Sentence.ord)
        ).all()
        return SectionOut(
            id=sec.id,
            document_id=sec.document_id,
            heading_path=list(sec.heading_path or []),
            anchor=sec.anchor,
            text=sec.text,
            sentences=[
                SentenceOut(
                    id=r.id,
                    ord=r.ord,
                    text=r.text,
                    kind=r.kind,
                    char_start=r.char_start,
                    char_end=r.char_end,
                )
                for r in rows
            ],
        )
