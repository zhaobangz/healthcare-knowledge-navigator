from __future__ import annotations

import math

import numpy as np
import pytest

from evals.common import GoldTarget, matches
from evals.stats import (
    aurc,
    auroc,
    brier_score,
    expected_calibration_error,
    fit_logistic,
    mrr,
    ndcg_at_k,
    paired_bootstrap_delta,
    recall_at_k,
    risk_coverage,
)


def test_retrieval_metrics_hand_worked() -> None:
    # Two gold targets; ranks: miss, target0, target0 again, target1.
    ranked = [set(), {0}, {0}, {1}]
    assert recall_at_k(ranked, 2, 2) == 0.5
    assert recall_at_k(ranked, 2, 4) == 1.0
    assert mrr(ranked) == 0.5
    # Gains only for newly matched targets: ranks 2 and 4.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, 2, 10) == pytest.approx(dcg / idcg)
    assert mrr([set(), set()]) == 0.0


def test_gold_matching_is_case_insensitive_substring_within_source() -> None:
    gold = [GoldTarget("cdc-sti-2021", "latent syphilis › treatment")]
    assert matches(gold, "cdc-sti-2021", ["Syphilis", "Latent Syphilis", "Treatment"]) == {0}
    assert matches(gold, "other-doc", ["Syphilis", "Latent Syphilis", "Treatment"]) == set()


def test_calibration_metrics() -> None:
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [True, True, False, False]
    assert brier_score(conf, correct) == pytest.approx(0.01)
    assert expected_calibration_error(conf, correct, bins=2) == pytest.approx(0.1)
    assert auroc(conf, correct) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(auroc([0.2, 0.3], [True, True]))


def test_risk_coverage_and_aurc() -> None:
    # Most confident answer is wrong: risk 1.0 at 1/3 coverage.
    scores = [0.9, 0.5, 0.1]
    correct = [False, True, True]
    curve = risk_coverage(scores, correct)
    assert curve[0] == (pytest.approx(1 / 3), 1.0)
    assert curve[-1][1] == pytest.approx(1 / 3)
    assert aurc(scores, correct) == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)
    # A perfect ranking has lower AURC.
    assert aurc([0.1, 0.5, 0.9], correct) < aurc(scores, correct)


def test_paired_bootstrap_delta_detects_consistent_improvement() -> None:
    a = [0.2, 0.3, 0.4, 0.5, 0.6] * 4
    b = [x + 0.1 for x in a]
    d, lo, _hi = paired_bootstrap_delta(a, b)
    assert d == pytest.approx(0.1)
    assert lo > 0


def test_logistic_fit_is_nonnegative_and_separates() -> None:
    rng = np.random.default_rng(0)
    x_good = rng.uniform(0.6, 1.0, size=(40, 1))
    x_bad = rng.uniform(0.0, 0.4, size=(40, 1))
    noise = rng.uniform(0, 1, size=(80, 1))  # an uninformative feature
    x = np.hstack([np.vstack([x_good, x_bad]), noise])
    y = np.array([1] * 40 + [0] * 40)
    w, _b = fit_logistic(x, y)
    assert (w >= 0).all()
    assert w[0] > w[1]
