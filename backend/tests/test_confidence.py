from __future__ import annotations

from conftest import make_source
from hkn.ask import Prepared, finalize
from hkn.confidence.evidence import CitedDoc, evidence_strength
from hkn.confidence.facts_check import check_claim
from hkn.confidence.nli import NliScore
from hkn.confidence.scorer import Weights, compute_features, score_reliability
from hkn.generation.answer import AnswerBlock, CollectedAnswer
from hkn.generation.citations import MappedCitation
from hkn.generation.prompts import ABSTAIN_PREFIX
from hkn.retrieval.gates import GateResult, gate_answer, gate_retrieval
from hkn.retrieval.pipeline import Candidate


def prior_weights() -> Weights:
    from hkn.confidence.scorer import PRIOR_BANDS, PRIOR_BIAS, PRIOR_WEIGHTS

    return Weights(dict(PRIOR_WEIGHTS), PRIOR_BIAS, dict(PRIOR_BANDS), calibrated=False)


def cite(source_index: int, sentence_ids: list[int]) -> MappedCitation:
    return MappedCitation(source_index, 10, 1, sentence_ids, "", "title", "url")


def answer(*blocks: tuple[str, list[int]]) -> CollectedAnswer:
    return CollectedAnswer(
        blocks=[AnswerBlock(text, [cite(0, ids)] if ids else []) for text, ids in blocks]
    )


def fake_relevance(question, texts):
    return [6.0 for _ in texts]


def fake_nli(entail: float, contradiction: float = 0.0):
    def fn(pairs):
        return [NliScore(entail, 1 - entail - contradiction, contradiction) for _ in pairs]

    return fn


def test_facts_check_catches_a_drifted_dose() -> None:
    cited = "Ceftriaxone 500 mg IM in a single dose for persons weighing <150 kg"
    assert check_claim("Give ceftriaxone 500 mg IM once.", cited).score == 1.0
    drifted = check_claim("Give ceftriaxone 250 mg IM once.", cited)
    assert drifted.missing == ["250"]
    assert drifted.score == 0.5
    wrong_drug = check_claim("Give cefixime 500 mg.", cited)
    assert "cefixime" in wrong_drug.missing


def test_facts_check_normalizes_thousands_separators() -> None:
    assert check_claim("2,400,000 units", "2400000 units").score == 1.0


def test_features_and_reliability_monotonic_in_support(gonorrhea_sources) -> None:
    a = answer(
        ("CDC recommends ", []),
        ("ceftriaxone 500 mg IM in a single dose", [101]),
    )
    kw = {"question": "First-line gonorrhea treatment?", "relevance_fn": fake_relevance}
    strong = compute_features(
        a, gonorrhea_sources, top_rerank_logit=6.0, nli_fn=fake_nli(0.95), **kw
    )
    weak = compute_features(a, gonorrhea_sources, top_rerank_logit=6.0, nli_fn=fake_nli(0.2), **kw)
    assert strong.relevance > 0.8
    assert strong.coverage == 1.0  # short connective block is not a claim
    assert strong.facts_ok == 1.0
    w = prior_weights()
    assert score_reliability(strong, w).score > score_reliability(weak, w).score
    assert score_reliability(strong, w).band == "High"
    assert not score_reliability(strong, w).calibrated


def test_uncited_substantive_block_lowers_coverage(gonorrhea_sources) -> None:
    a = answer(
        ("Use ceftriaxone 500 mg IM in a single dose", [101]),
        (". Patients should also be retested three months after treatment for reinfection.", []),
    )
    f = compute_features(a, gonorrhea_sources, top_rerank_logit=6.0, nli_fn=fake_nli(0.9))
    assert [c.status for c in f.claims] == ["supported", "uncited"]
    assert f.coverage == 0.5


def test_claims_span_blocks_and_pool_their_citations(gonorrhea_sources) -> None:
    seen_pairs: list[tuple[str, str]] = []

    def recording_nli(pairs):
        seen_pairs.extend(pairs)
        return fake_nli(0.9)(pairs)

    a = answer(
        ("CDC recommends ", []),
        ("ceftriaxone 500 mg IM in a single dose", [101]),
        (", or 1 g for persons weighing ≥150 kg", [102]),
        (".", []),
    )
    f = compute_features(a, gonorrhea_sources, top_rerank_logit=6.0, nli_fn=recording_nli)
    assert len(f.claims) == 1
    premise, hypothesis = seen_pairs[0]
    assert hypothesis.startswith("CDC recommends ceftriaxone 500 mg")
    # The premise carries the source title and both cited sentences.
    assert premise.startswith("CDC — Sexually Transmitted Infections")
    assert "1 g ceftriaxone" in premise and "500 mg IM" in premise
    assert f.facts_ok == 1.0


def test_contradiction_caps_band_at_low(gonorrhea_sources) -> None:
    a = answer(("ceftriaxone 500 mg IM in a single dose", [101]))
    f = compute_features(
        a, gonorrhea_sources, top_rerank_logit=9.0, nli_fn=fake_nli(0.3, contradiction=0.6)
    )
    rel = score_reliability(f, prior_weights())
    assert f.contradicted
    assert rel.band == "Low"
    assert any("contradict" in r for r in rel.reasons)


def test_gates_fail_closed(gonorrhea_sources) -> None:
    assert not gate_retrieval([], min_rerank_logit=-3, rerank_enabled=True).passed
    cand = Candidate(1, 10, 1, 0, 1, "t", "", "k", "T", "CDC", "guideline", 2021, [])
    cand.rerank_score = -6.0
    assert not gate_retrieval([cand], min_rerank_logit=-3, rerank_enabled=True).passed
    cand.rerank_score = None
    assert not gate_retrieval([cand], min_rerank_logit=-3, rerank_enabled=True).passed
    assert gate_retrieval([cand], min_rerank_logit=-3, rerank_enabled=False).passed

    assert not gate_answer(CollectedAnswer()).passed
    assert not gate_answer(answer((f"{ABSTAIN_PREFIX} pediatric dosing.", []))).passed
    assert not gate_answer(answer(("Ceftriaxone works.", []))).passed  # uncited
    assert not gate_answer(CollectedAnswer(error={"kind": "api"})).passed
    assert gate_answer(answer(("ceftriaxone 500 mg", [101]))).passed


def test_finalize_answered_and_insufficient(gonorrhea_sources) -> None:
    prepared = Prepared("q", None, None, None, gonorrhea_sources, GateResult(True))
    status, events, rel, evidence, _ = finalize(
        prepared,
        answer(("ceftriaxone 500 mg IM in a single dose", [101])),
        nli_fn=fake_nli(0.9),
        relevance_fn=fake_relevance,
    )
    assert status == "answered"
    assert [e["event"] for e in events] == ["confidence", "done"]
    assert rel is not None and evidence is not None and evidence["rating"] == "Strong"

    status, events, rel, _, reason = finalize(
        prepared,
        answer((f"{ABSTAIN_PREFIX} this.", [])),
        nli_fn=fake_nli(0.9),
        relevance_fn=fake_relevance,
    )
    assert status == "insufficient" and rel is None
    assert events[0]["event"] == "insufficient" and reason and reason.startswith(ABSTAIN_PREFIX)


def test_evidence_strength_is_separate_and_penalizes_old_or_superseded_sources() -> None:
    fresh = evidence_strength(
        [CitedDoc(1, "guideline", "CDC", "t", 2024, False)], current_year=2026
    )
    old_review = evidence_strength(
        [CitedDoc(2, "review", "X", "t", 2009, False)], current_year=2026
    )
    superseded = evidence_strength(
        [CitedDoc(3, "guideline", "CDC", "t", 2016, True)], current_year=2026
    )
    assert fresh.rating == "Strong"
    assert old_review.rating == "Limited"
    assert superseded.score < fresh.score
    two = evidence_strength(
        [CitedDoc(1, "rct", "A", "t", 2024, False), CitedDoc(2, "rct", "B", "t", 2023, False)],
        current_year=2026,
    )
    assert two.n_documents == 2
    assert make_source(0, ["x"]).doc_type == "guideline"


def test_questions_and_lead_ins_are_not_scored_as_claims(gonorrhea_sources) -> None:
    a = answer(
        ("Recommended regimens for adults:", []),
        ("\nWhat does the CDC recommend? ", [100]),
        ("Ceftriaxone 500 mg IM in a single dose.", [101]),
    )
    f = compute_features(a, gonorrhea_sources, top_rerank_logit=6.0, nli_fn=fake_nli(0.95))
    assert [c.status for c in f.claims] == ["connective", "connective", "supported"]
    assert f.nli_min == 0.95
