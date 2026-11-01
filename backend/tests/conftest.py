from __future__ import annotations

import json
from pathlib import Path

import pytest

from hkn.retrieval.expand import SourceSection, SourceSentence

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def make_source(
    index: int, sentences: list[str], *, first_id: int = 100, **overrides
) -> SourceSection:
    fields = {
        "index": index,
        "section_id": 10 + index,
        "document_id": 1 + index,
        "source_key": "cdc-sti-2021",
        "doc_title": "Sexually Transmitted Infections Treatment Guidelines, 2021",
        "org": "CDC",
        "doc_type": "guideline",
        "pub_year": 2021,
        "url": "https://www.cdc.gov/mmwr/volumes/70/rr/rr7004a1.htm",
        "license": "Public domain (CDC MMWR)",
        "authors": "",
        "superseded": False,
        "heading_path": ["Gonococcal Infections", "Uncomplicated Gonococcal Infection"],
        "anchor": None,
        "best_rerank": 6.0,
        "best_rank": index + 1,
    }
    fields.update(overrides)
    src = SourceSection(**fields)
    offset = 0
    for i, text in enumerate(sentences):
        src.sentences.append(
            SourceSentence(
                id=first_id + i,
                ord=i,
                text=text,
                kind="sentence",
                char_start=offset,
                char_end=offset + len(text),
                matched=True,
            )
        )
        offset += len(text) + 1
    return src


@pytest.fixture
def gonorrhea_sources() -> list[SourceSection]:
    return [
        make_source(
            0,
            [
                "Recommended Regimen for Uncomplicated Gonococcal Infection",
                "Ceftriaxone 500 mg IM in a single dose for persons weighing <150 kg",
                "* For persons weighing ≥150 kg, 1 g ceftriaxone should be administered.",
            ],
        )
    ]
