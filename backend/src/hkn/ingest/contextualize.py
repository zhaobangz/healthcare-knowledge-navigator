"""Contextual retrieval (concept 13): Claude writes a short situating context per chunk.

Following Anthropic's contextual-retrieval recipe, but with a bounded prompt: instead of
the whole document (some guidelines exceed 100k tokens), each request carries the
document title, its outline (top-level headings) and the chunk's local section window —
about 4k tokens at most, so the cost is predictable and caching is a bonus, not a
requirement. Requests go through the Message Batches API (half price).

The context is used for embeddings and full-text search only (the `llm` context mode);
citations always point at the original sentence text. A refused or failed request
leaves the chunk on its free heading-path prefix.

    hkn ingest contextualize --scope guidelines            # estimate only
    hkn ingest contextualize --scope guidelines --yes      # submit a batch
    hkn ingest contextualize --collect                     # poll, store, re-embed
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from hkn.config import Settings, get_settings
from hkn.ingest.chunk import approx_tokens
from hkn.llm import estimate_cost, get_client
from hkn.models import Chunk, Document, Section, Sentence

GUIDELINE_TYPES = ("guideline", "protocol", "drug_label")
WINDOW_TOKENS = 2500
MAX_OUTLINE = 60

CONTEXT_SYSTEM = (
    "You situate short passages from clinical documents so that a search engine can find "
    "them. Given a document outline, a local excerpt and one chunk from that excerpt, reply "
    "with one or two sentences that say what the chunk is about and where it sits in the "
    "document (condition, population, intervention, recommendation or label section). Reply "
    "with the context sentences only."
)


@dataclass
class ContextRequest:
    chunk_id: int
    prompt: str

    @property
    def est_input_tokens(self) -> int:
        return approx_tokens(CONTEXT_SYSTEM) + approx_tokens(self.prompt)


def _batch_state_path(settings: Settings) -> Path:
    return settings.data_dir / "runs" / "context_batches.json"


def build_requests(
    session: Session, *, scope: str, limit: int | None = None
) -> list[ContextRequest]:
    query = (
        select(Chunk, Section, Document)
        .join(Section, Section.id == Chunk.section_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.llm_context.is_(None))
        .order_by(Chunk.document_id, Chunk.section_id, Chunk.first_sentence_ord)
    )
    if scope == "guidelines":
        query = query.where(Document.doc_type.in_(GUIDELINE_TYPES))
    rows = session.execute(query.limit(limit) if limit else query).all()
    outlines: dict[int, str] = {}
    sentences: dict[int, list[Sentence]] = {}
    out: list[ContextRequest] = []
    for chunk, section, doc in rows:
        if doc.id not in outlines:
            heads = session.scalars(
                select(Section.heading_path)
                .where(Section.document_id == doc.id)
                .order_by(Section.ord)
            ).all()
            top = list(dict.fromkeys(h[0] for h in heads if h))[:MAX_OUTLINE]
            outlines[doc.id] = "\n".join(f"- {h}" for h in top)
        if section.id not in sentences:
            sentences[section.id] = list(
                session.scalars(
                    select(Sentence).where(Sentence.section_id == section.id).order_by(Sentence.ord)
                )
            )
        window = _window_text(
            sentences[section.id], chunk.first_sentence_ord, chunk.last_sentence_ord
        )
        prompt = (
            f"<document>{doc.org} — {doc.title} ({doc.pub_year or 'n.d.'})</document>\n"
            f"<outline>\n{outlines[doc.id]}\n</outline>\n"
            f"<section>{' › '.join(section.heading_path or [])}</section>\n"
            f"<excerpt>\n{window}\n</excerpt>\n"
            f"<chunk>\n{chunk.text}\n</chunk>"
        )
        out.append(ContextRequest(chunk.id, prompt))
    return out


def _window_text(sents: list[Sentence], first: int, last: int) -> str:
    chosen = [s for s in sents if first <= s.ord <= last]
    used = sum(approx_tokens(s.text) for s in chosen)
    lo, hi = first - 1, last + 1
    by_ord = {s.ord: s for s in sents}
    while used < WINDOW_TOKENS and (lo in by_ord or hi in by_ord):
        for o in (lo, hi):
            if o in by_ord and used + approx_tokens(by_ord[o].text) <= WINDOW_TOKENS:
                chosen.append(by_ord[o])
                used += approx_tokens(by_ord[o].text)
        lo, hi = lo - 1, hi + 1
    return " ".join(s.text for s in sorted(chosen, key=lambda s: s.ord))


def estimate(requests: list[ContextRequest], settings: Settings) -> float:
    tokens_in = sum(r.est_input_tokens for r in requests)
    # Adaptive thinking at low effort plus ~60 words of output.
    return estimate_cost(settings.context_model, tokens_in, 400 * len(requests), batch=True)


def submit(requests: list[ContextRequest], settings: Settings) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = get_client().messages.batches.create(
        requests=[
            Request(
                custom_id=f"chunk-{r.chunk_id}",
                params=MessageCreateParamsNonStreaming(
                    model=settings.context_model,
                    max_tokens=2000,
                    system=CONTEXT_SYSTEM,
                    output_config={"effort": settings.context_effort},  # type: ignore[typeddict-item]
                    messages=[{"role": "user", "content": r.prompt}],
                ),
            )
            for r in requests
        ]
    )
    path = _batch_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {"pending": []}
    state["pending"].append(
        {"id": batch.id, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(requests)}
    )
    path.write_text(json.dumps(state, indent=2))
    return batch.id


def collect(
    embedder: Callable[[list[str]], list[list[float]]],
    settings: Settings | None = None,
    *,
    poll_seconds: int = 60,
) -> dict[str, int]:
    """Wait for pending batches, store contexts, and re-embed the `llm` mode."""
    from hkn.db import session_scope
    from hkn.ingest.pipeline import embedding_text

    settings = settings or get_settings()
    path = _batch_state_path(settings)
    state = json.loads(path.read_text()) if path.exists() else {"pending": []}
    client = get_client()
    counts = {"stored": 0, "refused": 0, "errored": 0}
    for entry in list(state["pending"]):
        while (batch := client.messages.batches.retrieve(entry["id"])).processing_status != "ended":
            print(
                f"batch {entry['id']}: {batch.processing_status}, {batch.request_counts.processing} processing"
            )
            time.sleep(poll_seconds)
        contexts: dict[int, str] = {}
        for result in client.messages.batches.results(entry["id"]):
            chunk_id = int(result.custom_id.removeprefix("chunk-"))
            if result.result.type != "succeeded":
                counts["errored"] += 1
                continue
            message = result.result.message
            if message.stop_reason == "refusal":
                counts["refused"] += 1  # fallbacks are not available on Batches
                continue
            text = " ".join(b.text for b in message.content if b.type == "text").strip()
            if text:
                contexts[chunk_id] = text
        with session_scope() as session:
            chunks = session.scalars(select(Chunk).where(Chunk.id.in_(contexts))).all()
            for chunk in chunks:
                chunk.llm_context = contexts[chunk.id]
            vectors = embedder([embedding_text(c, "llm") for c in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding_llm = vector
            counts["stored"] += len(chunks)
        state["pending"].remove(entry)
        state.setdefault("done", []).append(entry)
        path.write_text(json.dumps(state, indent=2))
    return counts
