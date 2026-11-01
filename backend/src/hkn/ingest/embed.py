"""Local embeddings (concept 3) with BGE v1.5.

BGE is asymmetric: queries get a retrieval instruction prefix, passages do not. Vectors
are L2-normalized so pgvector's cosine distance and inner product agree.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from hkn.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.embed_model, device=settings.device)


def embed_passages(
    texts: list[str], *, batch_size: int = 32, show_progress: bool = False
) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    settings = get_settings()
    vector = _model().encode(
        settings.embed_query_instruction + text, normalize_embeddings=True, convert_to_numpy=True
    )
    return vector.tolist()


def embed_passage_query(text: str) -> list[float]:
    """Embed query-side text that reads like a passage (HyDE), so no instruction prefix."""
    return embed_passages([text])[0]
