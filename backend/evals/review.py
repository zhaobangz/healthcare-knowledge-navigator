"""Human review of answer labels, and judge–human agreement (Cohen's κ).

Shows each labeled answer with its key facts and the automatic label; press y to agree,
n to flip, s to skip, q to quit. Human labels are saved to data/runs/labels-human.jsonl
and always take precedence in calibration.

    python -m evals.review --against claude
"""

from __future__ import annotations

import argparse
import json
import textwrap

from evals.label import label_path, latest_run, load_labels, load_records


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(auto: dict[str, dict], human: dict[str, dict]) -> tuple[int, float]:
    common = sorted(set(auto) & set(human))
    return len(common), cohens_kappa(
        [auto[i]["correct"] for i in common], [human[i]["correct"] for i in common]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--against", choices=["claude", "keyfacts"], default="claude")
    ap.add_argument("--stats", action="store_true", help="only print agreement")
    args = ap.parse_args(argv)
    auto = load_labels(args.against)
    human = load_labels("human")
    if args.stats:
        n, kappa = agreement(auto, human)
        print(f"{n} records reviewed; Cohen's κ ({args.against} vs human) = {kappa:.2f}")
        return 0
    records = {r["id"]: r for r in load_records(latest_run())}
    todo = [i for i in auto if i in records and i not in human]
    print(f"{len(todo)} labels to review ({len(human)} already reviewed)")
    with label_path("human").open("a") as fh:
        for i, rid in enumerate(todo, 1):
            r, lab = records[rid], auto[rid]
            print("\n" + "=" * 88)
            print(f"[{i}/{len(todo)}] {r['qid']} · config {r['config']} · status {r['status']}")
            print(textwrap.fill(f"Q: {r['question']}", 88))
            print("Key facts:\n" + "\n".join(f"  - {f}" for f in r["key_facts"]))
            print(textwrap.fill(f"A: {r['answer'] or r['reason']}", 88))
            print(
                f"{args.against} label: {'CORRECT' if lab['correct'] else 'INCORRECT'} — {lab.get('rationale', '')}"
            )
            choice = input("agree? [y]es / [n]o, flip / [s]kip / [q]uit: ").strip().lower()
            if choice == "q":
                break
            if choice not in {"y", "n"}:
                continue
            correct = lab["correct"] if choice == "y" else not lab["correct"]
            fh.write(
                json.dumps(
                    {"id": rid, "labeler": "human", "correct": correct, "agreed_with": args.against}
                )
                + "\n"
            )
    n, kappa = agreement(auto, load_labels("human"))
    print(f"\nCohen's κ ({args.against} vs human) over {n} records: {kappa:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
