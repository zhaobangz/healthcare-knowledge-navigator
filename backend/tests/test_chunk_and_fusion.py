from __future__ import annotations

from hkn.ingest.chunk import ChunkSpan, approx_tokens, heading_prefix, pack_chunks
from hkn.retrieval.fusion import rrf_fuse, rrf_scores


def test_pack_chunks_respects_budget_and_overlap() -> None:
    sentences = ["x" * 400] * 5  # 100 tokens each
    spans = pack_chunks(sentences, target_tokens=250, overlap=1)
    # 2 sentences fit per chunk; each next chunk restarts on the previous chunk's last sentence.
    assert spans == [ChunkSpan(0, 1), ChunkSpan(1, 2), ChunkSpan(2, 3), ChunkSpan(3, 4)]
    for span in spans:
        assert sum(approx_tokens(s) for s in sentences[span.first : span.last + 1]) <= 250


def test_oversized_sentence_is_its_own_chunk_and_progress_is_guaranteed() -> None:
    sentences = ["short.", "y" * 4000, "tail."]
    spans = pack_chunks(sentences, target_tokens=50, overlap=1)
    assert ChunkSpan(1, 1) in spans
    assert spans[-1].last == 2
    firsts = [s.first for s in spans]
    assert firsts == sorted(set(firsts))  # strictly advancing


def test_pack_chunks_empty() -> None:
    assert pack_chunks([], target_tokens=250, overlap=1) == []


def test_heading_prefix() -> None:
    assert heading_prefix("CDC", "STI Guidelines", ["Syphilis", "Treatment"]) == (
        "CDC — STI Guidelines › Syphilis › Treatment"
    )
    assert heading_prefix("CDC", "STI Guidelines", []) == "CDC — STI Guidelines"


def test_rrf_hand_worked_example() -> None:
    # dense: a, b, c   fts: b, d
    scores = rrf_scores([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["a"] == 1 / 61
    assert scores["d"] == 1 / 62
    # A chunk both arms like beats a chunk one arm loves.
    assert rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60) == ["b", "a", "d", "c"]


def test_rrf_ties_are_deterministic_by_first_appearance() -> None:
    assert rrf_fuse([["x", "y"], ["y", "x"]]) == ["x", "y"]
