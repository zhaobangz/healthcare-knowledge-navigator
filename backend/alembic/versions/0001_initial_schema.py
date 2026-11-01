"""Initial schema: documents → sections → sentences, and chunks with one vector + FTS
index per context mode (plain / heading / llm).

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_key", sa.String(200), nullable=False),
        sa.Column("version", sa.String(200), nullable=False, server_default=""),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("org", sa.String(100), nullable=False),
        sa.Column("doc_type", sa.String(40), nullable=False),
        sa.Column(
            "specialty", postgresql.ARRAY(sa.String(60)), nullable=False, server_default="{}"
        ),
        sa.Column("pub_year", sa.Integer),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("license", sa.Text, nullable=False, server_default=""),
        sa.Column("authors", sa.Text, nullable=False, server_default=""),
        sa.Column("superseded", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("source_key", name="uq_documents_source_key"),
    )
    op.create_index("ix_documents_doc_type", "documents", ["doc_type"])
    op.create_index("ix_documents_pub_year", "documents", ["pub_year"])
    op.create_index("ix_documents_specialty", "documents", ["specialty"], postgresql_using="gin")

    op.create_table(
        "sections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("heading_path", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("anchor", sa.String(300)),
        sa.Column("text", sa.Text, nullable=False),
    )
    op.create_index("ix_sections_document_id", "sections", ["document_id"])

    op.create_table(
        "sentences",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "section_id",
            sa.Integer,
            sa.ForeignKey("sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("page", sa.Integer),
        sa.Column("kind", sa.String(20), nullable=False, server_default="sentence"),
    )
    op.create_index("ix_sentences_section_ord", "sentences", ["section_id", "ord"])
    op.create_index("ix_sentences_document_id", "sentences", ["document_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "section_id",
            sa.Integer,
            sa.ForeignKey("sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("first_sentence_ord", sa.Integer, nullable=False),
        sa.Column("last_sentence_ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("context_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("llm_context", sa.Text),
        sa.Column("embedding_plain", Vector(EMBED_DIM)),
        sa.Column("embedding_heading", Vector(EMBED_DIM)),
        sa.Column("embedding_llm", Vector(EMBED_DIM)),
        sa.Column(
            "tsv_plain",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
        sa.Column(
            "tsv_heading",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', context_prefix || ' ' || text)", persisted=True),
        ),
        sa.Column(
            "tsv_llm",
            postgresql.TSVECTOR,
            sa.Computed(
                "to_tsvector('english', coalesce(llm_context, context_prefix) || ' ' || text)",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_chunks_section_id", "chunks", ["section_id"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    for mode in ("plain", "heading", "llm"):
        op.create_index(f"ix_chunks_tsv_{mode}", "chunks", [f"tsv_{mode}"], postgresql_using="gin")
        op.execute(
            f"CREATE INDEX ix_chunks_embedding_{mode}_hnsw ON chunks "
            f"USING hnsw (embedding_{mode} vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("sentences")
    op.drop_table("sections")
    op.drop_table("documents")
