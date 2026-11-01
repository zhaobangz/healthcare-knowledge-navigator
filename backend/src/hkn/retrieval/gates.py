"""Abstention gates. Both fail CLOSED to "Insufficient evidence".

Gate 1 (before generation) refuses to ask the model anything when retrieval found nothing
relevant: a fluent answer over irrelevant passages is the failure mode that matters most
in a clinical tool, and no downstream check is cheaper than not generating it.

Gate 2 (after generation) refuses to show an answer that carries no citations or that
opens with the abstention sentence the system prompt requires — an uncited clinical
answer is unverifiable by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from hkn.config import BACKEND_ROOT, Settings
from hkn.generation.answer import CollectedAnswer
from hkn.generation.prompts import ABSTAIN_PREFIX
from hkn.retrieval.pipeline import Candidate


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None


def retrieval_threshold(settings: Settings) -> float:
    """τ₁ from backend/tuned_params.json (written by `evals.tune_gate` on the dev split)
    when present, else the permissive default in settings."""
    path = BACKEND_ROOT / "tuned_params.json"
    if path.exists():
        value = json.loads(path.read_text()).get("gate_min_rerank_logit")
        if isinstance(value, int | float):
            return float(value)
    return settings.gate_min_rerank_logit


def gate_retrieval(
    candidates: list[Candidate], *, min_rerank_logit: float, rerank_enabled: bool
) -> GateResult:
    if not candidates:
        return GateResult(False, "No passages in the knowledge base matched this question.")
    if rerank_enabled:
        top = max((c.rerank_score for c in candidates if c.rerank_score is not None), default=None)
        if top is None:
            return GateResult(False, "Retrieved passages could not be scored for relevance.")
        if top < min_rerank_logit:
            return GateResult(
                False,
                f"The best-matching passage scored {top:.2f} for relevance, below the "
                f"threshold of {min_rerank_logit:.2f}; the knowledge base likely does not cover this.",
            )
    return GateResult(True)


def gate_answer(answer: CollectedAnswer) -> GateResult:
    if answer.error is not None:
        return GateResult(False, "The answer could not be generated.")
    if answer.refused:
        return GateResult(False, "The model declined to answer this question.")
    text = answer.text.strip()
    if not text:
        return GateResult(False, "The model returned an empty answer.")
    if text.startswith(ABSTAIN_PREFIX):
        return GateResult(False, text)
    if not answer.citations:
        return GateResult(False, "The answer carried no citations, so it cannot be verified.")
    return GateResult(True)
