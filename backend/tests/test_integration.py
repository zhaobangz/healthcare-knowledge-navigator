"""Integration tests against a real Postgres + pgvector (the `hkn_test` database).

Models are replaced by deterministic fakes (hashing embedder, keyword reranker) so CI
never downloads weights; what is under test is the SQL — filters, the tsvector/HNSW
indexes, RRF over real result sets, soft-filter fallback, and small-to-big expansion.

    make db-up && make test-integration
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from hkn.db import get_engine, session_scope
from hkn.ingest.parsed import Block, ParsedDocument, ParsedSection
from hkn.ingest.pipeline import DocMeta, store_document
from hkn.retrieval import pipeline as retrieval_pipeline
from hkn.retrieval.expand import expand_to_sections
from hkn.retrieval.pipeline import RetrievalConfig, retrieve
from hkn.retrieval.plan_query import QueryPlan, SoftFilters
from hkn.retrieval.search import Filters, dense_search, fts_search

pytestmark = pytest.mark.integration

TEST_URL = os.environ.get(
    "HKN_TEST_DATABASE_URL", "postgresql+psycopg://hkn:hkn@localhost:5434/hkn_test"
)
BACKEND = Path(__file__).resolve().parents[1]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-hashed-words vectors: texts sharing words are close in cosine space."""
    out = []
    for t in texts:
        v = np.zeros(768)
        for word in t.lower().split():
            v[int(hashlib.md5(word.strip(".,;:").encode()).hexdigest(), 16) % 768] += 1.0
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


def _words(text: str) -> set[str]:
    stop = {"what", "is", "the", "for", "a", "of", "in", "to", "and"}
    return {w.strip(".,;:?!") for w in text.lower().split()} - stop


def fake_rerank(query: str, passages: list[str]) -> list[float]:
    q = _words(query)
    return [float(len(q & _words(p))) for p in passages]


@pytest.fixture(scope="module", autouse=True)
def database(monkeypatch_module):
    env = {**os.environ, "HKN_ALEMBIC_URL": TEST_URL}
    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "base"],
        cwd=BACKEND,
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND,
        env=env,
        check=True,
        capture_output=True,
    )
    monkeypatch_module.setenv("HKN_DATABASE_URL", TEST_URL)
    from hkn.config import get_settings

    get_settings.cache_clear()
    get_engine.cache_clear()
    monkeypatch_module.setattr(retrieval_pipeline, "embed_query", lambda q: fake_embed([q])[0])
    monkeypatch_module.setattr(
        retrieval_pipeline, "embed_passage_query", lambda q: fake_embed([q])[0]
    )
    monkeypatch_module.setattr("hkn.retrieval.rerank.rerank_scores", fake_rerank)
    with session_scope() as s:
        for meta, sections in CORPUS:
            parsed = ParsedDocument(
                sections=[
                    ParsedSection(h, None, [Block("paragraph", t) for t in ts])
                    for h, ts in sections
                ]
            )
            store_document(s, meta, parsed, sha256=meta.source_key, embedder=fake_embed)
    yield
    get_settings.cache_clear()
    get_engine.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


CORPUS = [
    (
        DocMeta(
            "gl-sti",
            "STI Guidelines",
            "CDC",
            "guideline",
            "https://x.gov/sti",
            "PD",
            ["infectious_disease"],
            2021,
        ),
        [
            (
                ["Gonococcal Infections", "Treatment"],
                [
                    "Ceftriaxone 500 mg IM in a single dose treats uncomplicated gonorrhea.",
                    "Persons weighing at least 150 kg receive 1 g ceftriaxone.",
                ],
            ),
            (
                ["Syphilis", "Treatment"],
                ["Benzathine penicillin G 2.4 million units IM treats primary syphilis."],
            ),
        ],
    ),
    (
        DocMeta(
            "label-cef",
            "Ceftriaxone label",
            "FDA/DailyMed",
            "drug_label",
            "https://x.gov/label",
            "PD",
            ["infectious_disease"],
            2026,
        ),
        [
            (
                ["DOSAGE AND ADMINISTRATION"],
                ["The usual adult daily dose of ceftriaxone is 1 to 2 grams given once a day."],
            )
        ],
    ),
    (
        DocMeta(
            "gl-crc",
            "Colorectal Cancer Screening",
            "USPSTF",
            "guideline",
            "https://x.gov/crc",
            "PD",
            ["oncology"],
            2021,
        ),
        [
            (
                ["Starting and Stopping Ages"],
                ["The USPSTF recommends colorectal cancer screening starting at age 45 years."],
            )
        ],
    ),
]


def test_schema_has_hnsw_and_gin_indexes() -> None:
    with session_scope() as s:
        names = {
            r[0]
            for r in s.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"))
        }
    assert {
        "ix_chunks_embedding_heading_hnsw",
        "ix_chunks_tsv_heading",
        "ix_chunks_tsv_plain",
    } <= names


def test_fts_or_query_matches_long_questions_and_respects_filters() -> None:
    terms = [
        "what",
        "is",
        "the",
        "recommended",
        "treatment",
        "for",
        "uncomplicated",
        "gonorrhea",
        "in",
        "adults",
    ]
    with session_scope() as s:
        hits = fts_search(s, terms, filters=Filters(), limit=5)
        labels_only = fts_search(
            s, ["ceftriaxone"], filters=Filters(doc_types=["drug_label"]), limit=5
        )
        assert hits, "an OR'd query must match even when not every word occurs"
        assert all(
            s.execute(
                text(
                    "SELECT d.doc_type FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=:i"
                ),
                {"i": h.chunk_id},
            ).scalar()
            == "drug_label"
            for h in labels_only
        )


def test_dense_search_with_year_filter() -> None:
    with session_scope() as s:
        hits = dense_search(
            s, fake_embed(["ceftriaxone dose"])[0], filters=Filters(year_min=2025), limit=5
        )
        years = {
            s.execute(
                text(
                    "SELECT d.pub_year FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=:i"
                ),
                {"i": h.chunk_id},
            ).scalar()
            for h in hits
        }
    assert years == {2026}


def test_retrieve_fuses_arms_and_drops_starving_soft_filters() -> None:
    plan = QueryPlan(
        is_clinical=True,
        asks_personal_advice=False,
        rewritten_query="ceftriaxone dose for uncomplicated gonorrhea",
        keywords=["ceftriaxone", "gonorrhea"],
        sub_questions=[],
        soft_filters=SoftFilters(specialties=["oncology"]),  # wrong on purpose: starves results
        hyde_passage="Ceftriaxone 500 mg IM is recommended.",
    )
    cfg = RetrievalConfig(top_k=4, soft_filter_min_hits=3)
    with session_scope() as s:
        result = retrieve(s, "What is the ceftriaxone dose for gonorrhea?", plan=plan, config=cfg)
        assert result.soft_filters_dropped
        assert result.candidates[0].source_key == "gl-sti"
        sources = expand_to_sections(s, result.candidates, window_tokens=1500)
    assert sources[0].heading_path == ["Gonococcal Infections", "Treatment"]
    assert sources[0].sentences[0].text.startswith("Ceftriaxone 500 mg IM")


def test_reingest_is_idempotent() -> None:
    meta, sections = CORPUS[2]
    parsed = ParsedDocument(
        sections=[ParsedSection(h, None, [Block("paragraph", t) for t in ts]) for h, ts in sections]
    )
    with session_scope() as s:
        assert (
            store_document(s, meta, parsed, sha256=meta.source_key, embedder=fake_embed).status
            == "unchanged"
        )


def test_informative_terms_orders_by_rarity_and_keeps_a_minimum() -> None:
    from hkn.retrieval.search import informative_terms

    with session_scope() as s:
        kept = informative_terms(
            s, ["ceftriaxone", "benzathine", "colorectal", "dose"], mode="plain", keep_min=2
        )
    # Nothing is under 5% df in a 4-chunk corpus, so the 2 rarest survive; "ceftriaxone"
    # appears in two chunks and must not be among them.
    assert len(kept) == 2 and "ceftriaxone" not in kept
