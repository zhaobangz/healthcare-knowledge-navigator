"""Cross-encoder reranking (concept 11).

Bi-encoder retrieval scores query and chunk independently; the cross-encoder reads them
together and is far better at "does this passage answer this question". Raw logits are
returned (identity activation): they are a ranking signal and a confidence *feature*, not
a probability — calibration happens downstream on labeled data.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from hkn.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(
        settings.rerank_model, device=settings.device, activation_fn=torch.nn.Identity()
    )


def rerank_scores(query: str, passages: list[str], *, batch_size: int = 16) -> list[float]:
    if not passages:
        return []
    scores = _model().predict(
        [(query, p) for p in passages], batch_size=batch_size, convert_to_numpy=True
    )
    return [float(s) for s in scores]
