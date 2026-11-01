"""Local NLI support check: does the cited text entail the claim?

This is the confidence signal that does not come from Claude. A DeBERTa-v3 cross-encoder
trained on SNLI/MultiNLI reads (premise = the cited sentences, hypothesis = the answer
block that cites them) and returns P(entailment), P(neutral), P(contradiction). It is
weaker than an LLM judge on long or paraphrastic claims, which is exactly why the
calibration step learns how much to trust it instead of hard-coding a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from hkn.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Label order for cross-encoder/nli-deberta-v3-base (see its model card).
LABELS = ("contradiction", "entailment", "neutral")


@dataclass(frozen=True)
class NliScore:
    entailment: float
    neutral: float
    contradiction: float


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(
        settings.nli_model, device=settings.device, activation_fn=torch.nn.Identity()
    )


def nli_scores(pairs: list[tuple[str, str]], *, batch_size: int = 8) -> list[NliScore]:
    """pairs: (premise, hypothesis)."""
    if not pairs:
        return []
    logits = np.asarray(
        _model().predict(pairs, batch_size=batch_size, convert_to_numpy=True), dtype=float
    )
    logits = logits.reshape(len(pairs), len(LABELS))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    idx = {label: i for i, label in enumerate(LABELS)}
    return [
        NliScore(
            entailment=float(p[idx["entailment"]]),
            neutral=float(p[idx["neutral"]]),
            contradiction=float(p[idx["contradiction"]]),
        )
        for p in probs
    ]
