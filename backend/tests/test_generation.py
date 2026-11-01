from __future__ import annotations

from conftest import load_fixture, make_source
from hkn.generation.answer import (
    AnswerEvent,
    CollectedAnswer,
    StreamTranslator,
    _final_events,
    build_request,
)
from hkn.generation.citations import CitationMapper, text_fragment_url
from hkn.generation.prompts import ABSTAIN_PREFIX, ANSWER_SYSTEM


def run_stream(events: list[dict], sources) -> tuple[list[AnswerEvent], CollectedAnswer]:
    translator = StreamTranslator(CitationMapper(sources))
    out: list[AnswerEvent] = []
    for raw in events:
        out.extend(translator.feed(raw))
    collected = CollectedAnswer()
    for e in out:
        collected.apply(e)
    return out, collected


def test_request_uses_one_search_result_block_per_section_and_one_text_block_per_sentence(
    gonorrhea_sources,
) -> None:
    req = build_request("First-line gonorrhea treatment?", gonorrhea_sources)
    content = req["messages"][0]["content"]
    assert content[0]["type"] == "search_result"
    assert content[0]["citations"] == {"enabled": True}
    assert [b["text"] for b in content[0]["content"]] == [
        s.text for s in gonorrhea_sources[0].sentences
    ]
    assert content[-1] == {"type": "text", "text": "First-line gonorrhea treatment?"}
    # API version 2026-10-21 rejects sampling parameters; citations reject structured outputs.
    for forbidden in ("temperature", "top_p", "top_k", "output_format"):
        assert forbidden not in req
    assert "format" not in req["output_config"]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["system"][0]["text"] == ANSWER_SYSTEM
    assert req["fallbacks"] == "default"
    assert req["thinking"] == {"type": "adaptive"}


def test_stream_translation_is_block_scoped_and_maps_citations_to_sentence_ids(
    gonorrhea_sources,
) -> None:
    fixture = load_fixture("answer_stream_basic.json")
    events, collected = run_stream(fixture["events"], gonorrhea_sources)
    types = [e.type for e in events]
    assert types[0] == "status" and events[0].data == {"phase": "thinking"}
    assert types[1] == "status" and events[1].data == {"phase": "writing"}
    # Citations arrive inside their block, between block_start and block_stop.
    first_citation = types.index("citation")
    assert types[first_citation - 2 : first_citation + 2] == [
        "block_start",
        "text",
        "citation",
        "block_stop",
    ]
    assert collected.text == (
        "For uncomplicated gonorrhea, CDC recommends ceftriaxone 500 mg IM in a single dose"
        ", or 1 g for persons weighing ≥150 kg"
    )
    assert [c.sentence_ids for c in collected.citations] == [[101], [102]]
    assert collected.blocks[0].citations == []  # connective text carries no citation
    assert "#:~:text=" in collected.citations[0].url


def test_out_of_range_citations_are_dropped_and_counted(gonorrhea_sources) -> None:
    mapper = CitationMapper(gonorrhea_sources)
    bad = [
        {
            "type": "search_result_location",
            "search_result_index": 3,
            "start_block_index": 0,
            "end_block_index": 1,
        },
        {
            "type": "search_result_location",
            "search_result_index": 0,
            "start_block_index": 2,
            "end_block_index": 9,
        },
        {"type": "char_location", "document_index": 0},
    ]
    assert all(mapper.map(c) is None for c in bad)
    assert mapper.invalid == 3


def test_fallback_block_becomes_served_by_and_text_continues(gonorrhea_sources) -> None:
    fixture = load_fixture("answer_stream_fallback_refusal.json")
    events, collected = run_stream(fixture["fallback_events"], gonorrhea_sources)
    served = [e for e in events if e.type == "served_by"]
    assert served and served[0].data == {
        "from_model": "claude-opus-5-5",
        "to_model": "claude-opus-5",
    }
    assert collected.text.endswith("Continued by the fallback model.")


def test_refusal_discards_partial_text(gonorrhea_sources) -> None:
    fixture = load_fixture("answer_stream_fallback_refusal.json")
    _events, collected = run_stream(fixture["fallback_events"], gonorrhea_sources)
    for e in _final_events(fixture["refusal_final"]):
        collected.apply(e)
    assert collected.refused and collected.refusal_category == "bio"
    assert collected.text == ""


def test_final_event_reports_usage() -> None:
    (done,) = _final_events(load_fixture("answer_stream_basic.json")["final"])
    assert done.type == "done"
    assert done.data["usage"]["input_tokens"] == 1800


def test_text_fragment_urls() -> None:
    assert text_fragment_url("https://x.gov/p", "Ceftriaxone 500 mg IM") == (
        "https://x.gov/p#:~:text=Ceftriaxone%20500%20mg%20IM"
    )
    assert text_fragment_url("https://x.gov/p#sec", "Treat now") == (
        "https://x.gov/p#sec:~:text=Treat%20now"
    )


def test_abstain_prefix_is_in_the_system_prompt() -> None:
    assert ABSTAIN_PREFIX in ANSWER_SYSTEM
    assert make_source(0, ["a"]).title.startswith("CDC — Sexually Transmitted")


def test_recorded_stream_if_present() -> None:
    """Runs against a real recording once `hkn record-fixture` has been run with a key."""
    import pytest

    from conftest import FIXTURES
    from hkn.retrieval.expand import SourceSection, SourceSentence

    path = FIXTURES / "answer_stream_recorded.json"
    if not path.exists():
        pytest.skip("no recorded stream yet (needs an API key: hkn record-fixture)")
    fixture = load_fixture(path.name)
    sources = []
    for s in fixture["sources"]:
        src = make_source(s["index"], [x["text"] for x in s["sentences"]])
        src.sentences = [
            SourceSentence(x["id"], i, x["text"], x["kind"], 0, 0, x["matched"])
            for i, x in enumerate(s["sentences"])
        ]
        sources.append(src)
    assert all(isinstance(s, SourceSection) for s in sources)
    _, collected = run_stream(fixture["events"], sources)
    for e in _final_events(fixture["final"]):
        collected.apply(e)
    assert collected.text and not collected.refused
    assert collected.citations, "a real answer over real sources must carry citations"


def test_extractive_mode_quotes_top_sentences_with_valid_citations(gonorrhea_sources) -> None:
    from hkn.generation.extractive import extractive_events

    def scorer(question, texts):  # prefer the regimen row
        return [5.0 if "500 mg" in t else 0.5 for t in texts]

    collected = CollectedAnswer()
    for e in extractive_events(
        "gonorrhea treatment?", gonorrhea_sources, max_sentences=1, scorer=scorer
    ):
        collected.apply(e)
    assert "Ceftriaxone 500 mg IM" in collected.text
    assert [c.sentence_ids for c in collected.citations] == [[101]]
    assert collected.done and collected.done["model"] == "extractive"
