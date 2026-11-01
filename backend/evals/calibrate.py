"""Fit and evaluate the reliability score.

1. Dataset: every *answered* record from an answers run, labeled correct/incorrect
   (human > claude > keyfacts per record).
2. Model: non-negative logistic regression on the confidence features.
3. Honest evaluation: 5-fold cross-validation grouped by question (all four
   configurations of a question land in the same fold), so no question is scored by a
   model that saw it. Metrics on the out-of-fold predictions: Brier, AUROC, ECE
   (equal-mass), AURC, each with a bootstrap CI, against two baselines — the prior
   hand-set weights and retrieval strength alone.
4. Bands: High / Moderate thresholds read off the out-of-fold risk–coverage curve
   (the lowest score at which selective risk stays ≤ 5% and ≤ 20%).
5. The final model is fitted on all labeled answers and written to
   confidence_weights.json, with the evaluation stored alongside it in `meta`.

Refuses to write weights with fewer than 30 labeled answers or a single class: a
"calibration" fitted on that would be a number with no meaning.

    python -m evals.calibrate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evals.common import RUNS_DIR
from evals.label import latest_run, load_labels, load_records
from evals.stats import (
    aurc,
    auroc,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    fit_logistic,
    reliability_table,
    risk_coverage,
    sigmoid,
)
from hkn.confidence.scorer import FEATURES, PRIOR_BIAS, PRIOR_WEIGHTS
from hkn.config import BACKEND_ROOT, REPO_ROOT

MIN_ANSWERS = 30
FIG_DIR = REPO_ROOT / "docs" / "figures"


def build_dataset(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = {name: load_labels(name) for name in ("human", "claude", "keyfacts")}
    rows, used = [], {"human": 0, "claude": 0, "keyfacts": 0}
    for r in records:
        if r["status"] != "answered" or not r.get("features"):
            continue
        for name, labels in sources.items():
            if r["id"] in labels:
                rows.append({**r, "correct": bool(labels[r["id"]]["correct"]), "labeler": name})
                used[name] += 1
                break
    return rows, used


def folds_by_question(rows: list[dict[str, Any]], k: int = 5) -> list[int]:
    qids = sorted({r["qid"] for r in rows})
    fold_of = {q: i % k for i, q in enumerate(qids)}
    return [fold_of[r["qid"]] for r in rows]


def oof_predictions(x: np.ndarray, y: np.ndarray, folds: list[int], k: int = 5) -> np.ndarray:
    preds = np.zeros(len(y))
    folds_arr = np.asarray(folds)
    for f in range(k):
        test = folds_arr == f
        if not test.any():
            continue
        train = ~test
        if len(set(y[train])) < 2:
            preds[test] = y[train].mean() if train.any() else 0.5
            continue
        w, b = fit_logistic(x[train], y[train])
        preds[test] = sigmoid(x[test] @ w + b)
    return preds


def band_thresholds(
    scores: np.ndarray, correct: np.ndarray, *, high_risk: float = 0.05, moderate_risk: float = 0.20
) -> dict[str, float]:
    order = np.argsort(-scores)
    s_sorted, err = scores[order], 1 - correct[order]
    cum_risk = np.cumsum(err) / np.arange(1, len(err) + 1)

    def lowest_threshold(max_risk: float, fallback: float) -> float:
        ok = np.where(cum_risk <= max_risk)[0]
        return float(s_sorted[ok.max()]) if ok.size else fallback

    high = lowest_threshold(high_risk, 0.95)
    moderate = min(lowest_threshold(moderate_risk, 0.75), high)
    return {"high": round(high, 4), "moderate": round(moderate, 4)}


def metric_block(scores: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    idx = np.arange(len(scores))

    def ci(fn) -> list[float]:
        point, lo, hi = bootstrap_ci(
            idx, stat=lambda a: fn(scores[a.astype(int)], correct[a.astype(int)]), n=1000
        )
        return [round(point, 4), round(lo, 4), round(hi, 4)]

    return {
        "brier": ci(lambda s, c: brier_score(s, c)),
        "auroc": ci(lambda s, c: auroc(s, c) if 0 < c.sum() < len(c) else math.nan),
        "ece": ci(lambda s, c: expected_calibration_error(s, c, bins=8)),
        "aurc": ci(lambda s, c: aurc(s, c)),
    }


# ------------------------------------------------------------------------------ figures


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="system-ui, sans-serif" font-size="12">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>'
    )


def reliability_svg(scores: np.ndarray, correct: np.ndarray) -> str:
    w, h, m = 360, 360, 44
    sx = lambda v: m + v * (w - 2 * m)  # noqa: E731
    sy = lambda v: h - m - v * (h - 2 * m)  # noqa: E731
    parts = [
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#7d908d" stroke-dasharray="4 4"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(0)}" stroke="#12312e"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(0)}" y2="{sy(1)}" stroke="#12312e"/>',
        f'<text x="{w / 2}" y="{h - 10}" text-anchor="middle" fill="#12312e">Predicted reliability</text>',
        f'<text x="14" y="{h / 2}" text-anchor="middle" fill="#12312e" transform="rotate(-90 14 {h / 2})">Observed accuracy</text>',
    ]
    for row in reliability_table(scores, correct, bins=8):
        r = 3 + 2 * math.sqrt(row["n"])
        parts.append(
            f'<circle cx="{sx(row["mean_confidence"]):.1f}" cy="{sy(row["accuracy"]):.1f}" r="{r:.1f}" fill="#0f6e66" fill-opacity="0.75"/>'
        )
    return _svg(w, h, "".join(parts))


def risk_coverage_svg(curves: dict[str, list[tuple[float, float]]]) -> str:
    w, h, m = 420, 320, 48
    sx = lambda v: m + v * (w - 2 * m)  # noqa: E731
    sy = lambda v: h - m - v * (h - 2 * m)  # noqa: E731
    colors = ["#0f6e66", "#a86a12", "#7d908d"]
    parts = [
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(0)}" stroke="#12312e"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(0)}" y2="{sy(1)}" stroke="#12312e"/>',
        f'<text x="{w / 2}" y="{h - 12}" text-anchor="middle" fill="#12312e">Coverage (share of answers shown)</text>',
        f'<text x="14" y="{h / 2}" text-anchor="middle" fill="#12312e" transform="rotate(-90 14 {h / 2})">Error rate among shown</text>',
    ]
    for i, (name, curve) in enumerate(curves.items()):
        pts = " ".join(f"{sx(c):.1f},{sy(r):.1f}" for c, r in curve)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colors[i % 3]}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{w - m - 4}" y="{m + 16 * i}" text-anchor="end" fill="{colors[i % 3]}">{name}</text>'
        )
    return _svg(w, h, "".join(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, default=None, help="answers-*.jsonl (default: latest)")
    ap.add_argument("--out", type=Path, default=BACKEND_ROOT / "confidence_weights.json")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write weights")
    args = ap.parse_args(argv)

    run = args.run or latest_run()
    rows, used = build_dataset(load_records(run))
    y = np.array([r["correct"] for r in rows], dtype=float)
    print(
        f"{len(rows)} labeled answers from {run.name} (labels: {used}); {int(y.sum()) if len(y) else 0} correct"
    )
    if len(rows) < MIN_ANSWERS or len(set(y)) < 2:
        print(
            f"insufficient data: need ≥{MIN_ANSWERS} labeled answers with both outcomes; nothing written"
        )
        return 1

    x = np.array([[r["features"][f] for f in FEATURES] for r in rows], dtype=float)
    folds = folds_by_question(rows)
    oof = oof_predictions(x, y, folds)
    prior = sigmoid(x @ np.array([PRIOR_WEIGHTS[f] for f in FEATURES]) + PRIOR_BIAS)
    retrieval_only = oof_predictions(x[:, [FEATURES.index("retrieval")]], y, folds)

    report = {
        "calibrated (out-of-fold)": metric_block(oof, y),
        "prior weights (uncalibrated)": metric_block(np.asarray(prior), y),
        "retrieval strength only": metric_block(retrieval_only, y),
    }
    bands = band_thresholds(oof, y)
    w, b = fit_logistic(x, y)
    labelers = sorted(k for k, v in used.items() if v)

    lines = [
        f"Reliability calibration — {len(rows)} answers, {len({r['qid'] for r in rows})} questions, labels: {', '.join(labelers)}",
        "",
    ]
    lines += ["| scorer | Brier ↓ | AUROC ↑ | ECE ↓ | AURC ↓ |", "|---|---|---|---|---|"]
    for name, m in report.items():
        cell = lambda v: "—" if math.isnan(v[0]) else f"{v[0]:.3f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
        lines.append(
            f"| {name} | {cell(m['brier'])} | {cell(m['auroc'])} | {cell(m['ece'])} | {cell(m['aurc'])} |"
        )
    lines += [
        "",
        f"Bands from the out-of-fold risk–coverage curve: High ≥ {bands['high']:.2f}, Moderate ≥ {bands['moderate']:.2f}",
    ]
    lines += [
        "Final weights: "
        + ", ".join(f"{f}={v:.2f}" for f, v in zip(FEATURES, w, strict=True))
        + f", bias={b:.2f}"
    ]
    if labelers == ["keyfacts"]:
        lines += [
            "",
            "WARNING: labels come only from the heuristic keyfacts labeler — a pipeline smoke test, not a calibration.",
        ]
    table = "\n".join(lines)
    print("\n" + table)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    real = not args.dry_run and labelers != ["keyfacts"]
    # Figures for the README only come from a real calibration; smoke tests stay in runs/.
    fig_dir = FIG_DIR if real else RUNS_DIR / f"calibration-figures-{stamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "reliability_diagram.svg").write_text(reliability_svg(oof, y))
    (fig_dir / "risk_coverage.svg").write_text(
        risk_coverage_svg(
            {
                "calibrated": risk_coverage(oof, y),
                "retrieval only": risk_coverage(retrieval_only, y),
                "prior weights": risk_coverage(np.asarray(prior), y),
            }
        )
    )
    (RUNS_DIR / f"calibration-{stamp}.md").write_text(table + "\n")
    if args.dry_run:
        return 0
    if labelers == ["keyfacts"]:
        print("not writing weights: heuristic labels only (use --labeler claude and human review)")
        return 0
    payload = {
        "weights": {f: round(float(v), 4) for f, v in zip(FEATURES, w, strict=True)},
        "bias": round(float(b), 4),
        "bands": bands,
        "meta": {
            "run": run.name,
            "n": len(rows),
            "labelers": used,
            "fitted": stamp,
            "evaluation": report,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
