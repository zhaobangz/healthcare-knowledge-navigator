"""Synthesize eval questions from corpus sections, then review them by hand.

Synthesized questions leak the wording of the passage they came from, which flatters
full-text search and HyDE. Mitigations, all measured rather than assumed:
* the model is told to ask as a clinician would, in its own words, never copying a run
  of more than three words from the passage;
* every question's lexical overlap with its gold section is stored and reported next to
  the hand-written set (`run_retrieval` splits all metrics by source);
* a question counts only after a human accepts it (`--review`); unreviewed ones are
  ignored by every eval.

    python -m evals.synthesize --n 60            # estimate
    python -m evals.synthesize --n 60 --yes      # generate (paid)
    python -m evals.synthesize --review          # accept / reject interactively
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from evals.common import QUESTIONS_DIR
from hkn.config import get_settings
from hkn.db import session_scope
from hkn.llm import estimate_cost, get_client
from hkn.models import Document, Section

SYNTH_PATH = QUESTIONS_DIR / "synth.jsonl"
_SKIP_HEADINGS = re.compile(
    r"method|background|introduction|conclusion|contributor|reference|acknowledg|search strategy",
    re.I,
)


class SynthQuestion(BaseModel):
    usable: bool = Field(
        description="False if the passage holds no clinically actionable fact worth asking about."
    )
    question: str = Field(description="A question a clinician would ask, in your own words.")
    key_facts: list[str] = Field(
        description="1-3 short facts from the passage that a correct answer must state, with exact numbers."
    )


SYNTH_SYSTEM = """You write evaluation questions for a clinical search system. Given one \
passage from a clinical guideline, drug label or paper, write a single question that a \
clinician might ask and that this passage answers. Ask it in your own words, the way a busy \
clinician would phrase it: do not copy any run of more than three consecutive words from the \
passage, and do not mention the document or section. Then list the key facts a correct \
answer must contain, copying numbers, units, drugs and populations exactly. If the passage \
contains no actionable clinical fact, set usable to false."""


def content_tokens(text: str) -> set[str]:
    stop = set(
        [
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "for",
            "and",
            "or",
            "with",
            "on",
            "by",
            "is",
            "are",
            "be",
            "what",
            "which",
            "how",
            "should",
            "when",
            "who",
            "does",
            "do",
            "at",
            "as",
            "from",
        ]
    )
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stop and len(t) > 2}


def lexical_overlap(question: str, passage: str) -> float:
    q = content_tokens(question)
    return 0.0 if not q else len(q & content_tokens(passage)) / len(q)


def sample_sections(n: int, seed: int = 7) -> list[tuple[Section, Document]]:
    with session_scope() as s:
        rows = s.execute(
            select(Section, Document)
            .join(Document, Document.id == Section.document_id)
            .where(func.length(Section.text) > 500)
        ).all()
        rows = [
            (sec, doc)
            for sec, doc in rows
            if not any(_SKIP_HEADINGS.search(h) for h in (sec.heading_path or []))
        ]
        s.expunge_all()
    rng = random.Random(seed)
    by_doc: dict[int, list] = {}
    for sec, doc in rows:
        by_doc.setdefault(doc.id, []).append((sec, doc))
    # Round-robin across documents so one long guideline cannot dominate.
    pools = [rng.sample(v, len(v)) for v in by_doc.values()]
    rng.shuffle(pools)
    picked: list[tuple[Section, Document]] = []
    while len(picked) < n and any(pools):
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return picked


def generate(n: int, allow_spend: bool) -> None:
    settings = get_settings()
    sections = sample_sections(n)
    est = estimate_cost(settings.judge_model, 1800 * len(sections), 600 * len(sections))
    print(f"{len(sections)} sections → questions on {settings.judge_model}: ~${est:.2f}")
    if not allow_spend:
        sys.exit("refusing to spend without --yes")
    existing = (
        [json.loads(line) for line in SYNTH_PATH.read_text().splitlines()]
        if SYNTH_PATH.exists()
        else []
    )
    next_id = len(existing) + 1
    with SYNTH_PATH.open("a") as fh:
        for sec, doc in sections:
            passage = sec.text[:6000]
            response = get_client().messages.parse(
                model=settings.judge_model,
                max_tokens=3000,
                system=SYNTH_SYSTEM,
                output_config={"effort": "low"},  # type: ignore[typeddict-item]
                messages=[{"role": "user", "content": f"Passage ({doc.org}):\n{passage}"}],
                output_format=SynthQuestion,
            )
            q = response.parsed_output
            if q is None or not q.usable:
                continue
            qid = f"s{next_id:03d}"
            row = {
                "id": qid,
                "split": "dev" if next_id % 2 else "test",
                "source": "synth",
                "question": q.question,
                "gold": [{"source_key": doc.source_key, "heading": (sec.heading_path or [""])[-1]}],
                "key_facts": q.key_facts,
                "lexical_overlap": round(lexical_overlap(q.question, sec.text), 3),
                "reviewed": False,
            }
            fh.write(json.dumps(row) + "\n")
            next_id += 1
            print(f"{qid} overlap={row['lexical_overlap']:.2f}  {q.question}")


def review() -> None:
    rows = [json.loads(line) for line in SYNTH_PATH.read_text().splitlines()]
    for row in rows:
        if row.get("reviewed"):
            continue
        print("\n" + "=" * 88)
        print(textwrap.fill(f"{row['id']}: {row['question']}", 88))
        print(
            f"gold: {row['gold'][0]['source_key']} › {row['gold'][0]['heading']}  overlap {row['lexical_overlap']:.2f}"
        )
        print("facts:\n" + "\n".join(f"  - {f}" for f in row["key_facts"]))
        choice = input("[a]ccept / [r]eject / [s]kip / [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "a":
            row["reviewed"] = True
        elif choice == "r":
            row["reviewed"] = "rejected"
    SYNTH_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    accepted = sum(r.get("reviewed") is True for r in rows)
    print(f"{accepted} accepted of {len(rows)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args(argv)
    if args.review:
        review()
    else:
        generate(args.n, args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
