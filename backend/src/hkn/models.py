"""ORM models.

The sentence is the unit everything else hangs off: chunks are runs of consecutive
sentences, the answer model sees sentences as individual citable blocks, and a citation's
block range maps straight back to sentence rows. That keeps every offset in the system in
characters of one stored `sections.text` string, so no tokenizer offset arithmetic is
ever needed to highlight a cited span.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBED_DIM = 768  # must match Settings.embed_dim and the migration

DOC_TYPES = ("guideline", "protocol", "systematic_review", "rct", "review", "drug_label", "other")
SENTENCE_KINDS = ("sentence", "list_item", "table_row", "caption")


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source_key", name="uq_documents_source_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_key: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(200), default="")
    title: Mapped[str] = mapped_column(Text)
    org: Mapped[str] = mapped_column(String(100))
    doc_type: Mapped[str] = mapped_column(String(40), index=True)
    specialty: Mapped[list[str]] = mapped_column(ARRAY(String(60)), default=list)
    pub_year: Mapped[int | None] = mapped_column(Integer, index=True)
    url: Mapped[str] = mapped_column(Text)
    license: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[str] = mapped_column(Text, default="")
    superseded: Mapped[bool] = mapped_column(Boolean, default=False)
    sha256: Mapped[str] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sections: Mapped[list[Section]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Section.ord"
    )


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    anchor: Mapped[str | None] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="sections")
    sentences: Mapped[list[Sentence]] = relationship(
        back_populates="section", cascade="all, delete-orphan", order_by="Sentence.ord"
    )


class Sentence(Base):
    __tablename__ = "sentences"
    __table_args__ = (Index("ix_sentences_section_ord", "section_id", "ord"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    section_id: Mapped[int] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20), default="sentence")

    section: Mapped[Section] = relationship(back_populates="sentences")


CONTEXT_MODES = ("plain", "heading", "llm")


class Chunk(Base):
    """Retrieval unit. Each context mode (the concept-13 ablation) has its own embedding
    and its own generated tsvector, so the eval ladder can compare "no prefix", "heading
    path prefix" and "Claude-written context" on identical chunks without re-ingesting.
    The `llm` columns fall back to the heading prefix for chunks that were never (or
    could not be) contextualized, so that mode is always complete."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    first_sentence_ord: Mapped[int] = mapped_column(Integer)
    last_sentence_ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    context_prefix: Mapped[str] = mapped_column(Text, default="")
    llm_context: Mapped[str | None] = mapped_column(Text)
    embedding_plain: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    embedding_heading: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    embedding_llm: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    tsv_plain: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    tsv_heading: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', context_prefix || ' ' || text)", persisted=True),
    )
    tsv_llm: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(llm_context, context_prefix) || ' ' || text)",
            persisted=True,
        ),
    )
