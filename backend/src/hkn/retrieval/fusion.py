"""Reciprocal Rank Fusion (concept 7, hybrid search).

Rank-based rather than score-based on purpose: cosine similarity and ts_rank_cd live on
incomparable scales (bounded vs unbounded, corpus-dependent), and any learned mixing
weight would overfit a small eval set. With k=60 (Cormack et al., 2009) the gap between
ranks 1 and 2 is flattened, so a chunk both arms like beats a chunk one arm loves.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf_scores[K: Hashable](rankings: Sequence[Sequence[K]], *, k: int = 60) -> dict[K, float]:
    """RRF(d) = Σ_r 1 / (k + rank_r(d)), ranks 1-based. A list may not repeat an id."""
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_fuse[K: Hashable](rankings: Sequence[Sequence[K]], *, k: int = 60) -> list[K]:
    """Fused order, best first; ties broken by first appearance for determinism."""
    scores = rrf_scores(rankings, k=k)
    first_seen: dict[K, int] = {}
    for ranking in rankings:
        for key in ranking:
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
