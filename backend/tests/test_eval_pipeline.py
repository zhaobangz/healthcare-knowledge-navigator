from __future__ import annotations

import numpy as np
import pytest

from evals.calibrate import band_thresholds, folds_by_question, oof_predictions
from evals.label import fact_present, keyfacts_label
from evals.review import cohens_kappa
from evals.tune_gate import choose_threshold, rates


def test_fact_present_requires_exact_numbers() -> None:
    answer = "CDC recommends ceftriaxone 500 mg IM as a single dose."
    assert fact_present("ceftriaxone 500 mg IM as a single dose", answer)
    assert not fact_present("ceftriaxone 250 mg IM as a single dose", answer)
    assert not fact_present("doxycycline 100 mg twice daily for 7 days", answer)


def test_answering_a_should_refuse_question_is_incorrect() -> None:
    record = {"expected": "abstain", "status": "answered", "answer": "x", "key_facts": []}
    assert keyfacts_label(record)["correct"] is False
    assert keyfacts_label({**record, "status": "insufficient"})["correct"] is True


def test_cohens_kappa() -> None:
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(
        0.0
    )


def test_folds_keep_every_configuration_of_a_question_together() -> None:
    rows = [{"qid": q} for q in ["a", "a", "b", "c", "b", "d", "e", "f"]]
    folds = folds_by_question(rows, k=3)
    by_q: dict[str, set[int]] = {}
    for row, f in zip(rows, folds, strict=True):
        by_q.setdefault(row["qid"], set()).add(f)
    assert all(len(fs) == 1 for fs in by_q.values())


def test_oof_predictions_rank_an_informative_feature() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(60, 1))
    y = (x[:, 0] + rng.normal(0, 0.1, 60) > 0.5).astype(float)
    preds = oof_predictions(x, y, [i % 5 for i in range(60)])
    assert np.corrcoef(preds, y)[0, 1] > 0.5


def test_band_thresholds_from_risk_coverage() -> None:
    scores = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    bands = band_thresholds(scores, correct, high_risk=0.0, moderate_risk=0.2)
    assert bands["high"] == 0.7  # every answer scoring ≥ 0.7 was correct
    assert bands["moderate"] <= bands["high"]


def test_gate_threshold_separates_refuse_from_answerable() -> None:
    answerable = [3.0, 5.0, 6.0]
    refuse = [-4.0, -1.0, 4.5]
    tau = choose_threshold(answerable, refuse)
    tpr, fpr = rates(tau, answerable, refuse)
    assert fpr == 0.0 and tpr == pytest.approx(2 / 3)
