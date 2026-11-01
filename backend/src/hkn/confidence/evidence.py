"""Evidence strength — a display-only, GRADE-inspired rating of the *sources*.

Kept separate from answer reliability on purpose. Reliability asks "is this answer a
faithful reading of its sources?"; evidence strength asks "how authoritative and current
are those sources?". Folding authority into a correctness probability would let a
guideline citation inflate the confidence of a misread, so this rating never enters the
calibrated score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DOC_TYPE_WEIGHT = {
    "guideline": 1.0,
    "protocol": 0.9,  # curated treatment summaries (NCI)
    "systematic_review": 0.85,
    "drug_label": 0.85,  # authoritative for dosing and contraindications
    "rct": 0.7,
    "review": 0.5,
    "other": 0.4,
}
DOC_TYPE_LABEL = {
    "guideline": "clinical guideline",
    "protocol": "treatment summary",
    "systematic_review": "systematic review",
    "drug_label": "FDA drug label",
    "rct": "randomized trial",
    "review": "narrative review",
    "other": "research article",
}


def recency_factor(pub_year: int | None, current_year: int) -> float:
    if pub_year is None:
        return 0.8
    age = current_year - pub_year
    if age <= 5:
        return 1.0
    if age <= 10:
        return 0.85
    return 0.7


@dataclass
class CitedDoc:
    document_id: int
    doc_type: str
    org: str
    title: str
    pub_year: int | None
    superseded: bool


@dataclass
class EvidenceStrength:
    rating: str  # Strong | Moderate | Limited
    score: float
    n_documents: int
    reasons: list[str] = field(default_factory=list)


def evidence_strength(docs: list[CitedDoc], *, current_year: int) -> EvidenceStrength:
    unique = {d.document_id: d for d in docs}.values()
    if not unique:
        return EvidenceStrength("Limited", 0.0, 0, ["No sources were cited."])
    reasons: list[str] = []
    weights = []
    for d in unique:
        w = DOC_TYPE_WEIGHT.get(d.doc_type, 0.4) * recency_factor(d.pub_year, current_year)
        if d.superseded:
            w *= 0.5
            reasons.append(f"{d.org} {d.pub_year or ''}: superseded edition".strip())
        weights.append(w)
        reasons.append(
            f"{d.org} {DOC_TYPE_LABEL.get(d.doc_type, d.doc_type)} ({d.pub_year or 'n.d.'})"
        )
    n = len(weights)
    score = min(1.0, max(weights) + 0.05 * (n - 1))
    rating = "Strong" if score >= 0.85 else "Moderate" if score >= 0.65 else "Limited"
    if n > 1:
        reasons.append(f"{n} independent documents cited")
    return EvidenceStrength(rating, round(score, 3), n, reasons)
