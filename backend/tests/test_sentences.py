from __future__ import annotations

import pytest

from hkn.ingest.parsed import Block, ParsedSection
from hkn.ingest.sentences import build_section_text, split_sentences


def sents(text: str) -> list[str]:
    return [text[a:b] for a, b in split_sentences(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Administer ceftriaxone 0.5 mg/kg IM. Follow up in 7 days.",
            ["Administer ceftriaxone 0.5 mg/kg IM.", "Follow up in 7 days."],
        ),
        (
            "Use doxycycline, e.g. for chlamydia, vs. azithromycin. Retest at 3 months.",
            ["Use doxycycline, e.g. for chlamydia, vs. azithromycin.", "Retest at 3 months."],
        ),
        (
            "Doses are discussed in Sect. 4.2 of the label. Stop if rash occurs.",
            ["Doses are discussed in Sect. 4.2 of the label.", "Stop if rash occurs."],
        ),
        (
            "U.S. adults aged ≥18 yrs. should be screened. Repeat annually.",
            ["U.S. adults aged ≥18 yrs. should be screened.", "Repeat annually."],
        ),
        (
            "The 9-year IDFS rate was 83.3% (P = .26).[Level of evidence B1] Levamisole did not help.",
            [
                "The 9-year IDFS rate was 83.3% (P = .26).[Level of evidence B1]",
                "Levamisole did not help.",
            ],
        ),
    ],
)
def test_split_sentences_clinical_edge_cases(text: str, expected: list[str]) -> None:
    assert sents(text) == expected


def test_spans_exclude_surrounding_whitespace() -> None:
    text = "  First sentence.   Second one.  "
    for a, b in split_sentences(text):
        assert not text[a].isspace() and not text[b - 1].isspace()


def test_section_text_offsets_round_trip() -> None:
    section = ParsedSection(
        heading_path=["Gonococcal Infections"],
        blocks=[
            Block("paragraph", "Ceftriaxone is first line. It is given IM."),
            Block("caption", "Recommended Regimen"),
            Block("table_row", "Ceftriaxone 500 mg IM in a single dose"),
            Block("list_item", "Retest in 3 months. Treat partners."),
        ],
    )
    text, records = build_section_text(section)
    assert [r.kind for r in records] == [
        "sentence",
        "sentence",
        "caption",
        "table_row",
        "list_item",
        "list_item",
    ]
    for r in records:
        assert text[r.char_start : r.char_end] == r.text
    # Table rows are atomic even when they contain sentence punctuation.
    assert records[3].text == "Ceftriaxone 500 mg IM in a single dose"
