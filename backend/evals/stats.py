"""Evaluation statistics: retrieval metrics, bootstrap CIs, calibration and selective
prediction. Pure Python + numpy so every number in the README is reproducible and
unit-tested against hand-worked values (tests/test_eval_stats.py).

Conventions stated out loud, because a metric quoted without them is not a number:
* Retrieval relevance is target-level: a question has 1+ gold targets (document +
  heading), and a retrieved chunk earns gain only for a target not already matched.
* ECE uses equal-mass bins (quantile edges) by default; equal-width is available.
* AURC is the area under the risk–coverage curve when answers are ranked by confidence,
  computed as the mean selective risk over all coverage levels (lower is better).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Floats = Sequence[float] | np.ndarray
Bools = Sequence[bool] | np.ndarray

# --------------------------------------------------------------------------- retrieval


def target_hits(matches: Sequence[set[int]]) -> list[int | None]:
    """For each rank, the first newly matched gold target index (or None).

    `matches[i]` is the set of gold-target indices the rank-(i+1) result matches.
    """
    seen: set[int] = set()
    out: list[int | None] = []
    for m in matches:
        new = sorted(m - seen)
        if new:
            seen.add(new[0])
            out.append(new[0])
        else:
            out.append(None)
    return out


def recall_at_k(matches: Sequence[set[int]], n_targets: int, k: int) -> float:
    if n_targets == 0:
        return math.nan
    found: set[int] = set().union(*matches[:k]) if matches[:k] else set()
    return len(found) / n_targets


def mrr(matches: Sequence[set[int]], k: int = 10) -> float:
    for rank, m in enumerate(matches[:k], start=1):
        if m:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(matches: Sequence[set[int]], n_targets: int, k: int = 10) -> float:
    if n_targets == 0:
        return math.nan
    gains = [1.0 if hit is not None else 0.0 for hit in target_hits(matches[:k])]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_targets, k)))
    return dcg / ideal


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    values: Floats,
    *,
    stat: Callable[[np.ndarray], float] = lambda a: float(np.mean(a)),
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap over items."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boots = np.array([stat(arr[row]) for row in idx])
    return stat(arr), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_delta(
    a: Floats, b: Floats, *, n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean(b - a) with a paired bootstrap CI (same questions resampled for both)."""
    diffs = [y - x for x, y in zip(a, b, strict=True) if not (math.isnan(x) or math.isnan(y))]
    return bootstrap_ci(diffs, n=n, alpha=alpha, seed=seed)


# --------------------------------------------------------------------------- calibration


def _equal_mass_edges(conf: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, bins + 1)))
    if edges.size == 1:
        edges = np.array([edges[0], edges[0]])
    return edges


def reliability_table(
    confidences: Floats,
    correct: Bools,
    *,
    bins: int = 10,
    scheme: str = "equal-mass",
) -> list[dict[str, float]]:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return []
    if scheme == "equal-mass":
        edges = _equal_mass_edges(conf, bins)
    elif scheme == "equal-width":
        edges = np.linspace(0, 1, bins + 1)
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}")
    # Right-closed bins; the first bin also includes its lower edge.
    which = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = which == b
        if mask.any():
            rows.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "n": int(mask.sum()),
                    "mean_confidence": float(conf[mask].mean()),
                    "accuracy": float(y[mask].mean()),
                }
            )
    return rows


def expected_calibration_error(
    confidences: Floats,
    correct: Bools,
    *,
    bins: int = 10,
    scheme: str = "equal-mass",
) -> float:
    rows = reliability_table(confidences, correct, bins=bins, scheme=scheme)
    n = sum(r["n"] for r in rows)
    if n == 0:
        return math.nan
    return sum(r["n"] / n * abs(r["accuracy"] - r["mean_confidence"]) for r in rows)


def brier_score(confidences: Floats, correct: Bools) -> float:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((conf - y) ** 2)) if conf.size else math.nan


def auroc(scores: Floats, labels: Bools) -> float:
    """Probability a random positive outscores a random negative (ties count 1/2)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def risk_coverage(scores: Floats, correct: Bools) -> list[tuple[float, float]]:
    """(coverage, selective risk) when answering only the top-c fraction by confidence."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-s, kind="stable")
    errors = 1.0 - y[order]
    n = s.size
    return [(k / n, float(errors[:k].mean())) for k in range(1, n + 1)]


def aurc(scores: Floats, correct: Bools) -> float:
    curve = risk_coverage(scores, correct)
    return float(np.mean([risk for _, risk in curve])) if curve else math.nan


# --------------------------------------------------------------------------- logistic fit


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    nonnegative: bool = True,
    l2: float = 1e-2,
    iterations: int = 3000,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Logistic regression by projected gradient descent.

    Non-negative weights encode the prior that every feature is a "more support ⇒ more
    reliable" signal; at the small n of a hand-built eval set this constraint is what
    keeps the fit from learning an inverted, overfit relationship. Features should be
    standardized or bounded in [0, 1].
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    w = np.zeros(x.shape[1])
    b = 0.0
    n = max(1, x.shape[0])
    for _ in range(iterations):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = x.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
        if nonnegative:
            w = np.maximum(w, 0.0)
    return w, b


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(z)))
