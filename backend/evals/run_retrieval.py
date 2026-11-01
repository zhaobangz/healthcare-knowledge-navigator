"""Retrieval ablation ladder (concept 15).

Each rung adds one technique to the previous rung and is evaluated on the same questions,
so the delta column (paired bootstrap, 95% CI) answers "did this technique help *here*?".
Negative and null results are reported as such.

Rungs that need a Claude query plan run only with `--with-plan --yes`; plans are cached
per question under data/runs/plans/, so the paid call happens once and reruns are free.

    python -m evals.run_retrieval                     # free rungs only
    python -m evals.run_retrieval --with-plan --yes   # + query plan, + HyDE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass

from evals.common import RUNS_DIR, Question, load_questions, matches
from evals.plans import get_plans
from evals.stats import bootstrap_ci, mrr, ndcg_at_k, paired_bootstrap_delta, recall_at_k
from hkn.db import session_scope
from hkn.retrieval.pipeline import RetrievalConfig, retrieve
from hkn.retrieval.plan_query import QueryPlan

EVAL_K = 10


@dataclass(frozen=True)
class Rung:
    name: str
    config: RetrievalConfig
    needs_plan: bool = False
    needs_llm_context: bool = False


def ladder() -> list[Rung]:
    base = RetrievalConfig(
        use_query_plan=False, use_hyde=False, use_sub_questions=False, rerank=False,
        context_mode="plain", top_k=EVAL_K,
    )  # fmt: skip
    hybrid = base.with_(dense=True, fts=True)
    return [
        Rung("dense (BGE)", base.with_(fts=False)),
        Rung("FTS, all terms OR'd", base.with_(dense=False, idf_prune=False)),
        Rung("FTS, IDF-pruned terms", base.with_(dense=False)),
        Rung("hybrid (RRF)", hybrid),
        Rung("+ rerank", hybrid.with_(rerank=True)),
        Rung("+ heading prefix", hybrid.with_(rerank=True, context_mode="heading")),
        Rung(
            "+ query plan",
            hybrid.with_(
                rerank=True, context_mode="heading", use_query_plan=True, use_sub_questions=True
            ),
            needs_plan=True,
        ),
        Rung(
            "+ HyDE",
            hybrid.with_(
                rerank=True,
                context_mode="heading",
                use_query_plan=True,
                use_sub_questions=True,
                use_hyde=True,
            ),
            needs_plan=True,
        ),
        Rung(
            "+ LLM context",
            hybrid.with_(
                rerank=True,
                context_mode="llm",
                use_query_plan=True,
                use_sub_questions=True,
                use_hyde=True,
            ),
            needs_plan=True,
            needs_llm_context=True,
        ),
    ]


def llm_context_available() -> bool:
    from sqlalchemy import func, select

    from hkn.models import Chunk

    with session_scope() as s:
        return bool(
            s.scalar(select(func.count()).select_from(Chunk).where(Chunk.llm_context.is_not(None)))
        )


# ------------------------------------------------------------------------------ run


def gold_overlap(q: Question) -> float:
    """Share of the question's content words found in its gold sections — the leakage
    measure that makes hand-written and synthesized results comparable."""
    from sqlalchemy import select

    from evals.synthesize import lexical_overlap
    from hkn.models import Document, Section

    with session_scope() as s:
        rows = s.execute(
            select(Section.text, Section.heading_path, Document.source_key)
            .join(Document, Document.id == Section.document_id)
            .where(Document.source_key.in_({g.source_key for g in q.gold}))
        ).all()
    text = " ".join(t for t, path, key in rows if matches(q.gold, key, list(path or [])))
    return lexical_overlap(q.question, text)


def by_source(result: dict, questions: list[Question]) -> str:
    overlap = {q.id: gold_overlap(q) for q in questions}
    lines = [
        f"By question source, final rung ({result['rung']}):",
        "",
        "| source | n | mean lexical overlap with gold | Recall@8 | nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for source in sorted({q["source"] for q in result["per_question"]}):
        rows = [q for q in result["per_question"] if q["source"] == source]
        recall = sum(r["recall@8"] for r in rows) / len(rows)
        ndcg = sum(r["ndcg@10"] for r in rows) / len(rows)
        ov = sum(overlap[r["id"]] for r in rows) / len(rows)
        lines.append(f"| {source} | {len(rows)} | {ov:.2f} | {recall:.3f} | {ndcg:.3f} |")
    return "\n".join(lines)


def evaluate_rung(rung: Rung, questions: list[Question], plans: dict[str, QueryPlan]) -> dict:
    per_q = []
    t0 = time.perf_counter()
    for q in questions:
        plan = plans.get(q.id) if rung.needs_plan else None
        with session_scope() as session:
            result = retrieve(session, q.question, plan=plan, config=rung.config)
        ranked = [matches(q.gold, c.source_key, c.heading_path) for c in result.candidates[:EVAL_K]]
        per_q.append(
            {
                "id": q.id,
                "source": q.source,
                "recall@8": recall_at_k(ranked, len(q.gold), 8),
                "mrr": mrr(ranked, EVAL_K),
                "ndcg@10": ndcg_at_k(ranked, len(q.gold), EVAL_K),
                "top": [
                    f"{c.source_key} › {' › '.join(c.heading_path)[-80:]}"
                    for c in result.candidates[:3]
                ],
            }
        )
    return {
        "rung": rung.name,
        "config": asdict(rung.config),
        "seconds": time.perf_counter() - t0,
        "per_question": per_q,
    }


def _fmt(point: float, lo: float, hi: float) -> str:
    return "—" if math.isnan(point) else f"{point:.3f} [{lo:.2f}, {hi:.2f}]"


def report(results: list[dict], n: int) -> str:
    metrics = ["recall@8", "mrr", "ndcg@10"]
    lines = [
        f"Retrieval ablation — {n} questions, 95% bootstrap CIs; Δ = paired change in nDCG@10 vs previous rung",
        "",
        "| rung | Recall@8 | MRR | nDCG@10 | Δ nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    prev = None
    for r in results:
        cols = [_fmt(*bootstrap_ci([q[m] for q in r["per_question"]])) for m in metrics]
        if prev is None:
            delta = ""
        else:
            d, lo, hi = paired_bootstrap_delta(
                [q["ndcg@10"] for q in prev["per_question"]],
                [q["ndcg@10"] for q in r["per_question"]],
            )
            sig = "" if lo <= 0 <= hi else " *"
            delta = f"{d:+.3f} [{lo:+.2f}, {hi:+.2f}]{sig}"
        lines.append(f"| {r['rung']} | {' | '.join(cols)} | {delta} |")
        prev = r
    lines.append("")
    lines.append("`*` = CI excludes zero.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--questions", default="hand,synth", help="comma-separated question files")
    ap.add_argument("--split", default=None, choices=["dev", "test"], help="default: all")
    ap.add_argument(
        "--with-plan", action="store_true", help="include rungs that need a Claude query plan"
    )
    ap.add_argument("--yes", action="store_true", help="allow paid API calls")
    ap.add_argument(
        "--rungs", default=None, help="comma-separated rung names to run (default: all eligible)"
    )
    args = ap.parse_args(argv)

    questions = [
        q
        for q in load_questions(*args.questions.split(","), split=args.split)
        if q.answerable and q.gold
    ]
    if not questions:
        sys.exit("no questions with gold targets found")
    rungs = [r for r in ladder() if args.with_plan or not r.needs_plan]
    if any(r.needs_llm_context for r in rungs) and not llm_context_available():
        print("skipping '+ LLM context': run `hkn ingest contextualize` first")
        rungs = [r for r in rungs if not r.needs_llm_context]
    if args.rungs:
        wanted = {name.strip() for name in args.rungs.split(",")}
        rungs = [r for r in rungs if r.name in wanted]
    plans = get_plans(questions, allow_spend=args.yes) if any(r.needs_plan for r in rungs) else {}

    results = []
    for rung in rungs:
        print(f"→ {rung.name}", flush=True)
        results.append(evaluate_rung(rung, questions, plans))
    table = report(results, len(questions)) + "\n\n" + by_source(results[-1], questions)
    print("\n" + table)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"retrieval-{stamp}.json"
    out.write_text(
        json.dumps({"questions": [q.id for q in questions], "results": results}, indent=1)
    )
    (RUNS_DIR / f"retrieval-{stamp}.md").write_text(table + "\n")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
