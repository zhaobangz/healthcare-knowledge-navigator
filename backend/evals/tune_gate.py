"""Tune gate 1 (minimum top rerank logit) on the dev split; report it on test.

Free: local models only. The threshold maximizes Youden's J (TPR on questions that
should be refused minus FPR on answerable ones) over dev questions, then is evaluated
once on the held-out test split. Writes backend/tuned_params.json, which the gate reads.

    python -m evals.tune_gate
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

from evals.common import Question, load_questions
from hkn.config import BACKEND_ROOT
from hkn.db import session_scope
from hkn.retrieval.pipeline import RetrievalConfig, retrieve

TUNED_PATH = BACKEND_ROOT / "tuned_params.json"


def top_logits(questions: list[Question], config: RetrievalConfig) -> list[float]:
    out = []
    for q in questions:
        with session_scope() as s:
            r = retrieve(s, q.question, config=config)
        out.append(max((c.rerank_score or -99.0) for c in r.candidates) if r.candidates else -99.0)
    return out


def rates(threshold: float, answerable: list[float], refuse: list[float]) -> tuple[float, float]:
    """(TPR: refuse-questions gated, FPR: answerable questions gated)."""
    tpr = sum(x < threshold for x in refuse) / max(1, len(refuse))
    fpr = sum(x < threshold for x in answerable) / max(1, len(answerable))
    return tpr, fpr


def choose_threshold(answerable: list[float], refuse: list[float]) -> float:
    candidates = sorted(set(answerable + refuse))
    # Midpoints between observed scores; ties broken toward the more permissive gate,
    # because refusing an answerable clinical question is cheaper than a wrong answer
    # only up to a point — the eval reports both rates either way.
    mids = [(a + b) / 2 for a, b in itertools.pairwise(candidates)] or candidates

    def youden(t: float) -> float:
        tpr, fpr = rates(t, answerable, refuse)
        return round(tpr - fpr, 9)  # round so float noise cannot break the tie rule

    best = max(mids, key=lambda t: (youden(t), -t))
    return round(best, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="hand,synth,unanswerable")
    ap.add_argument("--dry-run", action="store_true", help="do not write tuned_params.json")
    args = ap.parse_args(argv)
    config = RetrievalConfig.from_settings().with_(use_query_plan=False)
    names = args.questions.split(",")
    result = {}
    for split in ("dev", "test"):
        qs = load_questions(*names, split=split)
        ans = [q for q in qs if q.answerable]
        ref = [q for q in qs if not q.answerable]
        result[split] = {"answerable": top_logits(ans, config), "refuse": top_logits(ref, config)}
        print(f"{split}: {len(ans)} answerable, {len(ref)} should-refuse")
    dev, test = result["dev"], result["test"]
    tau = choose_threshold(dev["answerable"], dev["refuse"])
    for split, data in result.items():
        tpr, fpr = rates(tau, data["answerable"], data["refuse"])
        ans_sorted = sorted(data["answerable"])
        ref_sorted = sorted(data["refuse"])
        print(
            f"{split}: τ₁={tau:+.2f}  refuse-questions gated {tpr:.0%}  answerable gated {fpr:.0%}\n"
            f"  answerable top logits: min {ans_sorted[0]:+.2f} median {ans_sorted[len(ans_sorted) // 2]:+.2f}\n"
            f"  should-refuse top logits: {', '.join(f'{x:+.2f}' for x in ref_sorted)}"
        )
    if not args.dry_run:
        existing = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
        existing["gate_min_rerank_logit"] = tau
        existing["gate_fitted"] = {
            "split": "dev",
            "at": time.strftime("%Y-%m-%d"),
            "dev": dict(
                zip(("tpr", "fpr"), rates(tau, dev["answerable"], dev["refuse"]), strict=True)
            ),
            "test": dict(
                zip(("tpr", "fpr"), rates(tau, test["answerable"], test["refuse"]), strict=True)
            ),
        }
        TUNED_PATH.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"wrote {TUNED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
