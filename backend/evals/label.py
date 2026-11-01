"""Label answer records as correct / incorrect.

"Correct" = every hand-written key fact is stated, nothing contradicts the cited
sources, and (for false-premise questions) the premise is corrected. An answer to a
question that should have been refused is incorrect by definition.

Labelers:
    claude    a judge with its OWN rubric prompt (not the answer prompt, not the
              confidence features), so the label is not the model grading itself with
              the same instructions. Paid; prints an estimate and needs --yes.
    keyfacts  a deterministic token-overlap matcher. Free and weak: it exists so the
              calibration pipeline can be smoke-tested without an API key, and its
              output is stamped `labeler: keyfacts` everywhere it is used.

Human review (`python -m evals.review`) confirms or overrides either; calibration
prefers human > claude > keyfacts labels for each record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field

from evals.common import RUNS_DIR
from hkn.config import get_settings
from hkn.llm import estimate_cost, get_client, is_missing_credentials

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.I)
_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "and",
        "or",
        "by",
        "is",
        "are",
        "be",
        "as",
        "at",
        "not",
        "no",
        "that",
        "this",
        "from",
        "per",
        "its",
        "it",
    ]
)


def label_path(labeler: str) -> Path:
    return RUNS_DIR / f"labels-{labeler}.jsonl"


def load_labels(labeler: str) -> dict[str, dict[str, Any]]:
    path = label_path(labeler)
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_run(mode: str | None = None) -> Path:
    pattern = f"answers-{mode}-*.jsonl" if mode else "answers-*.jsonl"
    runs = sorted(RUNS_DIR.glob(pattern))
    if not runs:
        sys.exit("no answer runs found; run `python -m evals.run_answers` first")
    return runs[-1]


# ------------------------------------------------------------------------------ keyfacts


def fact_present(fact: str, answer: str, *, threshold: float = 0.7) -> bool:
    """Every number in the fact must appear in the answer, and ≥ threshold of its other
    content words."""
    fact_tokens = [t.lower() for t in _TOKEN.findall(fact) if t.lower() not in _STOP]
    answer_tokens = {t.lower() for t in _TOKEN.findall(answer)}
    numbers = [t for t in fact_tokens if t[0].isdigit()]
    words = [t for t in fact_tokens if not t[0].isdigit()]
    if any(n not in answer_tokens for n in numbers):
        return False
    if not words:
        return True
    return sum(w in answer_tokens for w in words) / len(words) >= threshold


def keyfacts_label(record: dict[str, Any]) -> dict[str, Any]:
    if record["expected"] == "abstain":
        correct = record["status"] != "answered"
        return {
            "correct": correct,
            "facts": [],
            "contradicted": False,
            "rationale": "should refuse",
        }
    facts = [{"fact": f, "present": fact_present(f, record["answer"])} for f in record["key_facts"]]
    contradicted = any(c["status"] == "conflict" for c in record.get("claims") or [])
    correct = (
        record["status"] == "answered" and all(f["present"] for f in facts) and not contradicted
    )
    return {
        "correct": correct,
        "facts": facts,
        "contradicted": contradicted,
        "rationale": "token overlap",
    }


# ------------------------------------------------------------------------------ claude judge


class FactVerdict(BaseModel):
    fact: str
    present: bool = Field(
        description="True if the answer states this fact (paraphrase is fine; numbers must match exactly)."
    )


class JudgeVerdict(BaseModel):
    facts: list[FactVerdict]
    contradicts_sources: bool = Field(
        description="True if any statement in the answer conflicts with the cited source text."
    )
    premise_corrected: bool | None = Field(
        description="For false-premise questions: does the answer correct the premise? Null otherwise."
    )
    rationale: str = Field(description="One or two sentences.")


JUDGE_SYSTEM = """You grade answers produced by a clinical question-answering system for an \
evaluation dataset. You are given the question, reference key facts written by a human, \
the system's answer, and the source passages the answer cited. Judge strictly and \
literally:
- A key fact is present only if the answer states it; paraphrase is acceptable, but every \
number, unit, drug and population must match the reference exactly.
- contradicts_sources is true if any statement in the answer conflicts with the cited \
passages (wrong dose, wrong population, reversed recommendation). Missing detail is not a \
contradiction.
- For a false-premise question, premise_corrected is true only if the answer explicitly \
corrects the false assumption. For other questions set it to null.
Do not reward fluency or length."""


def claude_label(record: dict[str, Any]) -> dict[str, Any]:
    if record["expected"] == "abstain":
        return keyfacts_label(record)  # no judgment needed: answering was the error
    settings = get_settings()
    cited = "\n".join(f"[{c['source']}] {c['cited_text']}" for c in record["citations"]) or "(none)"
    facts = "\n".join(f"- {f}" for f in record["key_facts"])
    false_premise = "yes" if record["expected"] == "correct_premise" else "no"
    content = (
        f"Question: {record['question']}\nFalse-premise question: {false_premise}\n\n"
        f"Reference key facts:\n{facts}\n\nAnswer:\n{record['answer'] or '(no answer: ' + str(record['reason']) + ')'}\n\n"
        f"Cited passages:\n{cited}"
    )
    response = get_client().messages.parse(
        model=settings.judge_model,
        max_tokens=4000,
        system=JUDGE_SYSTEM,
        output_config={"effort": settings.judge_effort},  # type: ignore[typeddict-item]
        messages=[{"role": "user", "content": content}],
        output_format=JudgeVerdict,
    )
    verdict = response.parsed_output
    if verdict is None:
        raise RuntimeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
    premise_ok = verdict.premise_corrected is not False
    correct = (
        record["status"] == "answered"
        and all(f.present for f in verdict.facts)
        and not verdict.contradicts_sources
        and premise_ok
    )
    return {
        "correct": correct,
        "facts": [f.model_dump() for f in verdict.facts],
        "contradicted": verdict.contradicts_sources,
        "rationale": verdict.rationale,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labeler", choices=["claude", "keyfacts"], default="claude")
    ap.add_argument("--run", type=Path, default=None, help="answers-*.jsonl (default: latest)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    run = args.run or latest_run()
    records = load_records(run)
    existing = load_labels(args.labeler)
    todo = [r for r in records if r["id"] not in existing]
    if args.labeler == "claude":
        settings = get_settings()
        n_paid = sum(r["expected"] != "abstain" for r in todo)
        est = estimate_cost(settings.judge_model, 3000 * n_paid, 800 * n_paid)
        print(f"{n_paid} judge calls on {settings.judge_model}: ~${est:.2f}")
        if not args.yes:
            sys.exit("refusing to spend without --yes")
    fn = claude_label if args.labeler == "claude" else keyfacts_label
    path = label_path(args.labeler)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for i, record in enumerate(todo, 1):
            try:
                label = fn(record)
            except anthropic.APIError as exc:
                print(f"  {record['id']}: judge failed ({exc}); skipped")
                continue
            except TypeError as exc:
                if is_missing_credentials(exc):
                    sys.exit("no Anthropic credentials configured")
                raise
            fh.write(json.dumps({"id": record["id"], "labeler": args.labeler, **label}) + "\n")
            print(
                f"[{i}/{len(todo)}] {record['id']}: {'correct' if label['correct'] else 'incorrect'}"
            )
    print(f"labels in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
