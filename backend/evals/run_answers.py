"""Answer eval: run every question under the full pipeline and three degraded ones.

Degraded configurations exist to manufacture real failures for calibration — with only
the full pipeline, a good system produces almost no wrong answers and a confidence model
fitted on them learns nothing:
    full            the production configuration
    top2            only 2 retrieved chunks (thin evidence)
    no_rerank       RRF order without the cross-encoder (noisier evidence)
    gold_withheld   the gold documents are filtered out (the answer is not in the sources)

Each record stores the answer, its citations, the confidence features and the outcome,
so labeling and calibration never re-run the model.

    python -m evals.run_answers --mode extractive          # free baseline
    python -m evals.run_answers --mode claude --yes        # paid; prints an estimate first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from evals.common import RUNS_DIR, Question, load_questions
from evals.plans import get_plans
from hkn.ask import ask
from hkn.config import get_settings
from hkn.db import session_scope
from hkn.llm import estimate_cost
from hkn.models import Document
from hkn.retrieval.pipeline import RetrievalConfig
from hkn.retrieval.search import Filters

CONFIGS = ("full", "top2", "no_rerank", "gold_withheld")
# Rough per-answer token use (8 windowed sections in, a cited answer plus adaptive
# thinking out); only used for the estimate printed before a paid run.
EST_IN, EST_OUT = 12_500, 2_500


@dataclass
class Job:
    question: Question
    config_name: str
    config: RetrievalConfig
    filters: Filters


def gold_doc_ids(q: Question) -> list[int]:
    keys = {g.source_key for g in q.gold}
    if not keys:
        return []
    with session_scope() as s:
        return list(s.scalars(select(Document.id).where(Document.source_key.in_(keys))))


def build_jobs(questions: list[Question], configs: list[str]) -> list[Job]:
    full = RetrievalConfig.from_settings()
    jobs: list[Job] = []
    for q in questions:
        for name in configs:
            if name == "gold_withheld" and not q.gold:
                continue
            cfg = {
                "full": full,
                "top2": full.with_(top_k=2),
                "no_rerank": full.with_(rerank=False),
            }.get(name, full)
            filters = (
                Filters(exclude_document_ids=gold_doc_ids(q))
                if name == "gold_withheld"
                else Filters()
            )
            jobs.append(Job(q, name, cfg, filters))
    return jobs


def run_job(job: Job, plan: Any, mode: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = ask(job.question.question, filters=job.filters, config=job.config, plan=plan)
    latency = time.perf_counter() - t0
    by_index = {s.index: s for s in result.sources}
    confidence = next((e["data"] for e in result.events if e["event"] == "confidence"), None)
    done = next((e["data"] for e in result.events if e["event"] == "done"), {})
    return {
        "qid": job.question.id,
        "question": job.question.question,
        "source": job.question.source,
        "split": job.question.split,
        "expected": job.question.expected,
        "category": job.question.category,
        "key_facts": job.question.key_facts,
        "config": job.config_name,
        "mode": mode,
        "status": result.status,
        "reason": result.reason,
        "answer": result.answer.text,
        "citations": [
            {"source": c.source_index + 1, "title": c.title, "cited_text": c.cited_text}
            for c in result.answer.citations
        ],
        "cited_source_keys": sorted(
            {
                by_index[c.source_index].source_key
                for c in result.answer.citations
                if c.source_index in by_index
            }
        ),
        "features": None if result.reliability is None else result.reliability.features,
        "reliability": None if result.reliability is None else result.reliability.score,
        "band": None if result.reliability is None else result.reliability.band,
        "evidence": None if result.evidence is None else result.evidence["rating"],
        "claims": None
        if confidence is None
        else [
            {k: c[k] for k in ("text", "status", "entailment", "facts_missing")}
            for c in confidence["claims"]
        ],
        "top_rerank": max(
            (s.best_rerank for s in result.sources if s.best_rerank is not None), default=None
        ),
        "latency_s": round(latency, 2),
        "usage": done.get("usage"),
    }


def summarize(records: list[dict[str, Any]]) -> str:
    lines = [
        "| config | n | answered | should-refuse questions refused | answerable questions answered | mean latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    for name in CONFIGS:
        rs = [r for r in records if r["config"] == name]
        if not rs:
            continue
        answered = sum(r["status"] == "answered" for r in rs)
        should_refuse = [r for r in rs if r["expected"] == "abstain"]
        refused_ok = sum(r["status"] != "answered" for r in should_refuse)
        answerable = [r for r in rs if r["expected"] != "abstain"]
        wrongly_refused = sum(r["status"] != "answered" for r in answerable)
        lat = sum(r["latency_s"] for r in rs) / len(rs)
        lines.append(
            f"| {name} | {len(rs)} | {answered} | {refused_ok}/{len(should_refuse)} | "
            f"{len(answerable) - wrongly_refused}/{len(answerable)} | {lat:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["claude", "extractive"], default="claude")
    ap.add_argument("--questions", default="hand,synth,unanswerable")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--no-plan", action="store_true", help="skip query plans (free retrieval)")
    ap.add_argument("--limit", type=int, default=None, help="first N jobs only")
    ap.add_argument("--yes", action="store_true", help="allow paid API calls")
    args = ap.parse_args(argv)

    settings = get_settings()
    settings.answer_mode = args.mode
    questions = load_questions(*args.questions.split(","), split=args.split)
    jobs = build_jobs(questions, [c.strip() for c in args.configs.split(",")])[: args.limit]
    if args.mode == "claude":
        est = estimate_cost(settings.answer_model, EST_IN * len(jobs), EST_OUT * len(jobs))
        print(f"{len(jobs)} answers on {settings.answer_model}: ~${est:.2f} (rough estimate)")
        if not args.yes:
            sys.exit("refusing to spend without --yes")
    plans = {} if args.no_plan else get_plans(questions, allow_spend=args.yes)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"answers-{args.mode}-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out.open("w") as fh:
        for i, job in enumerate(jobs, 1):
            cfg = job.config.with_(use_query_plan=not args.no_plan)
            record = run_job(
                Job(job.question, job.config_name, cfg, job.filters),
                plans.get(job.question.id),
                args.mode,
            )
            record["run"] = stamp
            record["id"] = f"{stamp}:{job.question.id}:{job.config_name}"
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            records.append(record)
            print(
                f"[{i}/{len(jobs)}] {job.question.id} {job.config_name}: {record['status']} {record['band'] or ''}",
                flush=True,
            )
    table = summarize(records)
    print("\n" + table + f"\n\nsaved {out}")
    (out.with_suffix(".md")).write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
