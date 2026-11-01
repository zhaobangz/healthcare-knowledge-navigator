"""Question sets and gold matching shared by the retrieval and answer evals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

QUESTIONS_DIR = Path(__file__).parent / "questions"
RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"


@dataclass
class GoldTarget:
    source_key: str
    heading: str  # case-insensitive substring of the section's heading path


@dataclass
class Question:
    id: str
    question: str
    split: str
    source: str  # "hand" | "synth" | "unanswerable"
    gold: list[GoldTarget] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    category: str | None = None  # unanswerable: off_topic | near_miss | false_premise
    expected: str = "answer"  # answer | abstain | correct_premise

    @property
    def answerable(self) -> bool:
        """False only for questions the system should refuse; a false-premise question
        is answerable (the right answer corrects the premise)."""
        return self.expected != "abstain"


def load_questions(*names: str, split: str | None = None) -> list[Question]:
    """Load question files by stem (hand, synth, unanswerable); filter by split."""
    out: list[Question] = []
    for name in names:
        path = QUESTIONS_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("reviewed", True) is not True:
                continue  # synthesized questions count only after human acceptance
            q = Question(
                id=raw["id"],
                question=raw["question"],
                split=raw.get("split", "dev"),
                source=raw.get("source", name),
                gold=[GoldTarget(**g) for g in raw.get("gold", [])],
                key_facts=list(raw.get("key_facts", [])),
                category=raw.get("category"),
                expected=raw.get("expected", "answer"),
            )
            if split is None or q.split == split:
                out.append(q)
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids across question files")
    return out


def matches(gold: list[GoldTarget], source_key: str, heading_path: list[str]) -> set[int]:
    """Indices of gold targets a retrieved chunk satisfies."""
    path = " › ".join(heading_path).lower()
    return {
        i for i, g in enumerate(gold) if g.source_key == source_key and g.heading.lower() in path
    }
