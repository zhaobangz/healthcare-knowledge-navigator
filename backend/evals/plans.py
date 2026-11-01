"""Query-plan cache shared by the evals: a plan is paid for once per (model, question)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from evals.common import RUNS_DIR, Question
from hkn.config import get_settings
from hkn.llm import estimate_cost
from hkn.retrieval.plan_query import QueryPlan, plan_query


def _cache_path(model: str) -> Path:
    return RUNS_DIR / "plans" / f"{model}.jsonl"


def qkey(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def load_plan_cache(model: str) -> dict[str, QueryPlan]:
    path = _cache_path(model)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["key"]] = QueryPlan.model_validate(row["plan"])
    return out


def get_plans(questions: list[Question], *, allow_spend: bool) -> dict[str, QueryPlan]:
    """Plans keyed by question id. Missing plans cost money: refuse without --yes."""
    settings = get_settings()
    cache = load_plan_cache(settings.plan_model)
    missing = [q for q in questions if qkey(q.question) not in cache]
    if missing:
        est = estimate_cost(settings.plan_model, 900 * len(missing), 700 * len(missing))
        print(
            f"query plans needed for {len(missing)} questions on {settings.plan_model}: ~${est:.2f}"
        )
        if not allow_spend:
            sys.exit("refusing to spend without --yes")
        path = _cache_path(settings.plan_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for q in missing:
                plan, error = plan_query(q.question)
                if plan is None:
                    print(f"  {q.id}: no plan ({error}); the raw question will be used")
                    continue
                cache[qkey(q.question)] = plan
                row = {"key": qkey(q.question), "question": q.question, "plan": plan.model_dump()}
                fh.write(json.dumps(row) + "\n")
    return {q.id: cache[qkey(q.question)] for q in questions if qkey(q.question) in cache}
