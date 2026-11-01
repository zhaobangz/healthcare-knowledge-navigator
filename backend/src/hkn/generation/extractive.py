"""Extractive answer mode: quote the best-matching source sentences verbatim.

Used when no language model is configured (`HKN_ANSWER_MODE=extractive`) and as the
baseline row in the answer eval. It never paraphrases, so it cannot misstate a dose, but
it also cannot synthesize across sources or say what is missing — which is exactly the
gap the eval should quantify. The UI labels it so it is never mistaken for a model answer.

It emits the same `AnswerEvent` stream as the Claude path, so gates, confidence scoring
and the UI are exercised identically.
"""

from __future__ import annotations

from collections.abc import Iterator

from hkn.generation.answer import AnswerEvent
from hkn.generation.citations import CitationMapper
from hkn.retrieval.expand import SourceSection

EXTRACTIVE_NOTICE = (
    "Extractive mode: no language model is configured, so this shows the best-matching "
    "source sentences quoted verbatim, not a written answer."
)
INTRO = "The most relevant passages in the sources:\n"


def extractive_events(
    question: str,
    sources: list[SourceSection],
    *,
    max_sentences: int = 4,
    max_sources: int = 3,
    min_score: float = 0.0,
    scorer=None,
) -> Iterator[AnswerEvent]:
    """`scorer(question, texts) -> list[float]` defaults to the cross-encoder reranker."""
    if scorer is None:
        from hkn.retrieval.rerank import rerank_scores as scorer

    candidates: list[tuple[int, int]] = []  # (source position, sentence position)
    for si, src in enumerate(sources[:max_sources]):
        for pi, sent in enumerate(src.sentences):
            if sent.kind != "caption" and len(sent.text.split()) >= 4:
                candidates.append((si, pi))
    # Score each sentence with its heading path: a regimen table row ("Ceftriaxone 500 mg
    # IM in a single dose") only matches the question in the context of its section.
    texts = [
        f"{' › '.join(sources[si].heading_path)}: {sources[si].sentences[pi].text}"
        for si, pi in candidates
    ]
    scores = scorer(question, texts) if texts else []
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
    chosen = sorted(c for c, s in ranked[:max_sentences] if s >= min_score)

    mapper = CitationMapper(sources)
    yield AnswerEvent("status", {"phase": "writing"})
    if not chosen:
        yield AnswerEvent("block_start", {"block": 0})
        yield AnswerEvent(
            "text", {"block": 0, "text": "The provided sources do not address this question."}
        )
        yield AnswerEvent("block_stop", {"block": 0})
        yield AnswerEvent("done", {"stop_reason": "end_turn", "model": "extractive", "usage": {}})
        return

    yield AnswerEvent("block_start", {"block": 0})
    yield AnswerEvent("text", {"block": 0, "text": INTRO})
    yield AnswerEvent("block_stop", {"block": 0})
    for n, (si, pi) in enumerate(chosen, start=1):
        src = sources[si]
        citation = mapper.map(
            {
                "type": "search_result_location",
                "search_result_index": src.index,
                "start_block_index": pi,
                "end_block_index": pi + 1,
                "cited_text": src.sentences[pi].text,
            }
        )
        yield AnswerEvent("block_start", {"block": n})
        yield AnswerEvent("text", {"block": n, "text": f"\n- {src.sentences[pi].text}"})
        if citation is not None:
            yield AnswerEvent("citation", {"block": n, **citation.to_dict()})
        yield AnswerEvent("block_stop", {"block": n})
    yield AnswerEvent("done", {"stop_reason": "end_turn", "model": "extractive", "usage": {}})
