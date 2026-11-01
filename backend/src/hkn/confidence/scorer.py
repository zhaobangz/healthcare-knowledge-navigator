"""Answer reliability: a calibrated P(answer is correct and supported by its citations).

The unit of assessment is the answer *sentence* (a claim), not the stream's text block:
Claude often splits one sentence into several blocks ("CDC recommends " + "ceftriaxone
500 mg IM [1]" + ", or 1 g for ≥150 kg [2]"), and a fragment is a poor NLI hypothesis.
Each claim is checked against the union of everything its overlapping blocks cite.

Features (all in [0, 1], all "more is better"):
  nli_min    min over cited claims of P(cited text entails claim)            [local NLI]
  facts_ok   share of numbers / drug names found verbatim in the cited text  [deterministic]
  retrieval  sigmoid of the top rerank logit                                 [local reranker]
  coverage   share of substantive claims that carry a citation               [structural]
  relevance  sigmoid of the cross-encoder score of (question, answer)        [local reranker]

The first four measure faithfulness; `relevance` is the only completeness-side signal
(does the answer respond to what was asked?). A faithful quote of the wrong passage
scores high on the first four and low on relevance.

None of the features asks Claude about Claude's own answer. The weights are fitted by
`evals/calibrate.py` (non-negative logistic regression, question-grouped CV) on labeled
answers and saved to confidence_weights.json; until that file exists, the hand-set prior
weights below are used and the result is labelled `calibrated: false` in the API, so an
uncalibrated number is never presented as a probability.

Hard rule outside the model: any claim the NLI model reads as contradicting its own
citation caps the band at Low, whatever the score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hkn.confidence.facts_check import check_claim
from hkn.config import get_settings
from hkn.generation.answer import CollectedAnswer
from hkn.ingest.sentences import split_sentences
from hkn.retrieval.expand import SourceSection

FEATURES = ("nli_min", "facts_ok", "retrieval", "coverage", "relevance")
PRIOR_WEIGHTS = {
    "nli_min": 3.0,
    "facts_ok": 1.5,
    "retrieval": 1.5,
    "coverage": 1.0,
    "relevance": 1.5,
}
PRIOR_BIAS = -4.5
PRIOR_BANDS = {"high": 0.80, "moderate": 0.55}
CONTRADICTION_CAP = 0.5  # P(contradiction) above which a claim counts as contradicted
SUBSTANTIVE_WORDS = 6  # uncited sentences shorter than this are connective text, not claims
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class ClaimAssessment:
    """One answer sentence. `start`/`end` are offsets into the full answer text, so the UI
    can mark each claim's support status in place."""

    start: int
    end: int
    text: str
    cited: bool
    substantive: bool
    source_indices: list[int] = field(default_factory=list)
    entailment: float | None = None
    contradiction: float | None = None
    facts_checked: int = 0
    facts_missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.cited:
            return "uncited" if self.substantive else "connective"
        if (self.contradiction or 0.0) > CONTRADICTION_CAP or self.facts_missing:
            return "conflict"
        return "supported" if (self.entailment or 0.0) >= 0.5 else "weak"


@dataclass
class Features:
    nli_min: float
    facts_ok: float
    retrieval: float
    coverage: float
    relevance: float
    contradicted: bool
    top_rerank_logit: float | None
    claims: list[ClaimAssessment]

    def vector(self) -> list[float]:
        return [getattr(self, name) for name in FEATURES]


@dataclass
class Reliability:
    score: float
    band: str  # High | Moderate | Low
    calibrated: bool
    features: dict[str, float]
    reasons: list[str]
    bands: dict[str, float] = field(default_factory=lambda: dict(PRIOR_BANDS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def claim_spans(text: str) -> list[tuple[int, int]]:
    """Answer sentences; list lines are split separately so each bullet is one claim."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.split("\n"):
        for a, b in split_sentences(line):
            spans.append((offset + a, offset + b))
        offset += len(line) + 1
    return spans


def compute_features(
    answer: CollectedAnswer,
    sources: list[SourceSection],
    *,
    top_rerank_logit: float | None,
    question: str = "",
    nli_fn: Any | None = None,
    relevance_fn: Any | None = None,
) -> Features:
    """`nli_fn(pairs) -> list[NliScore]` and `relevance_fn(question, [text]) -> [logit]`
    are injectable so tests run without the models."""
    if nli_fn is None:
        from hkn.confidence.nli import nli_scores as nli_fn
    if relevance_fn is None:
        from hkn.retrieval.rerank import rerank_scores as relevance_fn

    sentence_text = {s.id: s.text for src in sources for s in src.sentences}
    titles = {src.index: src.title for src in sources}

    # Lay the blocks out in one string so claims can be located across block boundaries.
    full = ""
    block_spans: list[tuple[int, int, int]] = []  # (start, end, block number)
    for i, block in enumerate(answer.blocks):
        block_spans.append((len(full), len(full) + len(block.text), i))
        full += block.text

    claims: list[ClaimAssessment] = []
    pairs: list[tuple[str, str]] = []
    pair_claims: list[int] = []
    for start, end in claim_spans(full):
        text = _BULLET.sub("", full[start:end]).strip()
        if not text:
            continue
        if text.endswith(("?", ":")):
            # Questions and lead-ins ("Recommended regimens:") assert nothing to verify.
            claims.append(ClaimAssessment(start, end, text, cited=False, substantive=False))
            continue
        cited: dict[int, list[int]] = {}  # source index → sentence ids, in citation order
        for b_start, b_end, i in block_spans:
            if b_start < end and start < b_end:
                for c in answer.blocks[i].citations:
                    ids = cited.setdefault(c.source_index, [])
                    ids.extend(sid for sid in c.sentence_ids if sid not in ids)
        evidence = [
            (idx, [sentence_text[sid] for sid in ids if sid in sentence_text])
            for idx, ids in cited.items()
        ]
        evidence = [(idx, texts) for idx, texts in evidence if texts]
        claim = ClaimAssessment(
            start=start,
            end=end,
            text=text,
            cited=bool(evidence),
            substantive=bool(evidence) or len(text.split()) >= SUBSTANTIVE_WORDS,
            source_indices=[idx for idx, _ in evidence],
        )
        if evidence:
            cited_plain = " ".join(t for _, texts in evidence for t in texts)
            fc = check_claim(text, cited_plain)
            claim.facts_checked, claim.facts_missing = fc.checked, fc.missing
            # Title + heading path in the premise lets "CDC recommends …" be entailed.
            premise = "\n".join(
                f"{titles.get(idx, '')}: {' '.join(texts)}" for idx, texts in evidence
            )
            pairs.append((premise, text))
            pair_claims.append(len(claims))
        claims.append(claim)

    for idx, score in zip(pair_claims, nli_fn(pairs), strict=True):
        claims[idx].entailment = score.entailment
        claims[idx].contradiction = score.contradiction

    answer_text = " ".join(c.text for c in claims if c.substantive)
    relevance = (
        _sigmoid(relevance_fn(question, [answer_text])[0] / 3.0)
        if question and answer_text
        else 0.0
    )
    cited_claims = [c for c in claims if c.cited]
    substantive = [c for c in claims if c.substantive]
    checked = sum(c.facts_checked for c in cited_claims)
    missing = sum(len(c.facts_missing) for c in cited_claims)
    return Features(
        nli_min=min((c.entailment or 0.0 for c in cited_claims), default=0.0),
        facts_ok=1.0 if checked == 0 else (checked - missing) / checked,
        retrieval=0.0 if top_rerank_logit is None else _sigmoid(top_rerank_logit / 3.0),
        coverage=0.0 if not substantive else sum(c.cited for c in substantive) / len(substantive),
        relevance=relevance,
        contradicted=any((c.contradiction or 0.0) > CONTRADICTION_CAP for c in cited_claims),
        top_rerank_logit=top_rerank_logit,
        claims=claims,
    )


@dataclass
class Weights:
    weights: dict[str, float]
    bias: float
    bands: dict[str, float]
    calibrated: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Weights:
        path = path or get_settings().confidence_weights_path
        if path.exists():
            raw = json.loads(path.read_text())
            return cls(raw["weights"], raw["bias"], raw["bands"], True, raw.get("meta", {}))
        return cls(dict(PRIOR_WEIGHTS), PRIOR_BIAS, dict(PRIOR_BANDS), False)


def score_reliability(features: Features, weights: Weights | None = None) -> Reliability:
    w = weights or Weights.load()
    z = w.bias + sum(w.weights.get(name, 0.0) * getattr(features, name) for name in FEATURES)
    score = _sigmoid(z)
    band = (
        "High"
        if score >= w.bands["high"]
        else "Moderate"
        if score >= w.bands["moderate"]
        else "Low"
    )
    reasons: list[str] = []
    if features.contradicted:
        band = "Low"
        reasons.append("A cited passage appears to contradict part of the answer.")
    if features.facts_ok < 1.0:
        missing = sorted({m for c in features.claims for m in c.facts_missing})
        reasons.append(f"Not found verbatim in the cited text: {', '.join(missing[:6])}.")
    if features.nli_min < 0.5:
        reasons.append("At least one cited statement is only weakly supported by its source.")
    if features.coverage < 1.0:
        reasons.append("Part of the answer is not tied to a citation.")
    if features.retrieval < 0.5:
        reasons.append("The best retrieved passage is only a moderate match for the question.")
    if features.relevance < 0.5:
        reasons.append("The answer may not address what was asked.")
    if not reasons:
        reasons.append("Every statement is cited and its citation supports it.")
    return Reliability(
        score=round(score, 4),
        band=band,
        calibrated=w.calibrated,
        features={name: round(getattr(features, name), 4) for name in FEATURES},
        reasons=reasons,
        bands=dict(w.bands),
    )


def preliminary_retrieval_strength(top_rerank_logit: float | None) -> dict[str, Any]:
    """Shown while the answer streams: a retrieval-only signal, explicitly not reliability."""
    if top_rerank_logit is None:
        return {"retrieval": None, "label": "unscored"}
    value = _sigmoid(top_rerank_logit / 3.0)
    label = "strong" if value >= 0.75 else "moderate" if value >= 0.4 else "weak"
    return {"retrieval": round(value, 3), "label": label}
