"""Ingest orchestration: parsed document → sections → sentences → chunks → embeddings.

Idempotent on `source_key` + content hash: an unchanged source is skipped, a changed one
replaces its old rows in a single transaction, so a failed re-ingest never leaves a
document half-updated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hkn.config import Settings, get_settings
from hkn.ingest.chunk import heading_prefix, pack_chunks
from hkn.ingest.fetch import fetch_cached
from hkn.ingest.parse_html import HtmlParserConfig, parse_html
from hkn.ingest.parse_jats import parse_jats
from hkn.ingest.parse_spl import parse_spl
from hkn.ingest.parsed import ParsedDocument
from hkn.ingest.sentences import build_section_text
from hkn.ingest.sources import Manifest, SourceSpec
from hkn.models import Chunk, Document, Section, Sentence

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class DocMeta:
    source_key: str
    title: str
    org: str
    doc_type: str
    url: str
    license: str
    specialty: list[str] = field(default_factory=list)
    pub_year: int | None = None
    version: str = ""
    authors: str = ""
    superseded: bool = False

    @classmethod
    def from_spec(cls, spec: SourceSpec) -> DocMeta:
        return cls(
            source_key=spec.source_key,
            title=spec.title,
            org=spec.org,
            doc_type=spec.doc_type,
            url=spec.link,
            license=spec.license,
            specialty=spec.specialty,
            pub_year=spec.pub_year,
            version=spec.version,
            superseded=spec.superseded,
        )


@dataclass
class IngestResult:
    source_key: str
    status: str  # "ingested" | "unchanged"
    sections: int = 0
    sentences: int = 0
    chunks: int = 0


def embedding_text(chunk: Chunk, mode: str) -> str:
    """The text embedded for a context mode; must mirror the tsv_* generated columns."""
    if mode == "plain":
        return chunk.text
    prefix = (chunk.llm_context or chunk.context_prefix) if mode == "llm" else chunk.context_prefix
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text


def store_document(
    session: Session,
    meta: DocMeta,
    parsed: ParsedDocument,
    *,
    sha256: str,
    embedder: Embedder | None,
    settings: Settings | None = None,
    force: bool = False,
) -> IngestResult:
    settings = settings or get_settings()
    existing = session.scalar(select(Document).where(Document.source_key == meta.source_key))
    if existing is not None:
        if existing.sha256 == sha256 and not force:
            return IngestResult(meta.source_key, "unchanged")
        session.delete(existing)
        session.flush()

    doc = Document(
        source_key=meta.source_key,
        version=meta.version,
        title=meta.title,
        org=meta.org,
        doc_type=meta.doc_type,
        specialty=meta.specialty,
        pub_year=meta.pub_year,
        url=meta.url,
        license=meta.license,
        authors=meta.authors,
        superseded=meta.superseded,
        sha256=sha256,
    )
    session.add(doc)
    session.flush()

    result = IngestResult(meta.source_key, "ingested")
    chunks: list[Chunk] = []
    for ord_, parsed_section in enumerate(parsed.sections):
        text, records = build_section_text(parsed_section)
        if not records:
            continue
        section = Section(
            document_id=doc.id,
            ord=ord_,
            heading_path=parsed_section.heading_path,
            anchor=parsed_section.anchor,
            text=text,
        )
        session.add(section)
        session.flush()
        for i, rec in enumerate(records):
            session.add(
                Sentence(
                    section_id=section.id,
                    document_id=doc.id,
                    ord=i,
                    text=rec.text,
                    char_start=rec.char_start,
                    char_end=rec.char_end,
                    page=rec.page,
                    kind=rec.kind,
                )
            )
        prefix = heading_prefix(meta.org, meta.title, parsed_section.heading_path)
        for span in pack_chunks(
            [r.text for r in records],
            target_tokens=settings.chunk_target_tokens,
            overlap=settings.chunk_overlap_sentences,
        ):
            chunks.append(
                Chunk(
                    section_id=section.id,
                    document_id=doc.id,
                    first_sentence_ord=span.first,
                    last_sentence_ord=span.last,
                    text=" ".join(r.text for r in records[span.first : span.last + 1]),
                    context_prefix=prefix,
                )
            )
        result.sections += 1
        result.sentences += len(records)

    if embedder is not None and chunks:
        embed_chunks(chunks, embedder)
    session.add_all(chunks)
    session.flush()
    result.chunks = len(chunks)
    return result


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> None:
    """Fill all three context-mode embeddings. Until a chunk is contextualized, its `llm`
    embedding is the heading one (same text), so it is copied rather than recomputed."""
    plain = embedder([embedding_text(c, "plain") for c in chunks])
    heading = embedder([embedding_text(c, "heading") for c in chunks])
    for chunk, p, h in zip(chunks, plain, heading, strict=True):
        chunk.embedding_plain = p
        chunk.embedding_heading = h
        chunk.embedding_llm = h if not chunk.llm_context else chunk.embedding_llm


def parse_source(spec: SourceSpec, raw: bytes) -> ParsedDocument:
    text = raw.decode("utf-8", errors="replace")
    if spec.kind == "html":
        return parse_html(text, HtmlParserConfig.from_manifest(spec.parser))
    if spec.kind == "spl":
        return parse_spl(text)
    if spec.kind == "jats":
        return parse_jats(text)
    raise ValueError(f"{spec.source_key}: unknown parser kind {spec.kind!r}")


def ingest_spec(
    session: Session,
    spec: SourceSpec,
    manifest: Manifest,
    *,
    embedder: Embedder | None,
    refresh: bool = False,
    force: bool = False,
    settings: Settings | None = None,
) -> IngestResult:
    settings = settings or get_settings()
    ext = "xml" if spec.kind in {"spl", "jats"} else "html"
    fetched = fetch_cached(
        spec.url,
        settings.raw_dir / f"{spec.source_key}.{ext}",
        user_agent=manifest.user_agent,
        refresh=refresh,
    )
    parsed = parse_source(spec, fetched.content)
    return store_document(
        session,
        DocMeta.from_spec(spec),
        parsed,
        sha256=fetched.sha256,
        embedder=embedder,
        settings=settings,
        force=force,
    )
