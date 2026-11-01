"use client";

import { Fragment } from "react";

import { blockOffsets, type AnswerState } from "@/lib/answerState";
import type { AnswerBlock, Claim } from "@/lib/types";

export interface ActiveCitation {
  sourceIndex: number;
  sentenceIds: number[];
}

interface Props {
  state: AnswerState;
  active: ActiveCitation | null;
  onCite: (c: ActiveCitation) => void;
}

const STATUS_CLASS: Partial<Record<Claim["status"], string>> = {
  weak: "claim-weak",
  conflict: "claim-conflict",
  uncited: "claim-uncited",
};

function claimTitle(claim: Claim): string | undefined {
  const sources = claim.source_indices.map((i) => i + 1).join(", ");
  switch (claim.status) {
    case "weak":
      return `Only weakly supported by source ${sources} (entailment ${(claim.entailment ?? 0).toFixed(2)}).`;
    case "conflict":
      return claim.facts_missing.length
        ? `Not found in source ${sources}: ${claim.facts_missing.join(", ")}.`
        : `Source ${sources} appears to contradict this statement.`;
    case "uncited":
      return "This statement has no citation.";
    default:
      return undefined;
  }
}

/** Split [start, end) at claim boundaries so each piece can carry its claim's mark. */
function pieces(text: string, start: number, claims: Claim[]) {
  const end = start + text.length;
  const cuts = new Set<number>([start, end]);
  for (const c of claims) {
    if (c.start > start && c.start < end) cuts.add(c.start);
    if (c.end > start && c.end < end) cuts.add(c.end);
  }
  const sorted = [...cuts].sort((a, b) => a - b);
  return sorted.slice(0, -1).map((from, i) => {
    const to = sorted[i + 1]!;
    const claim = claims.find((c) => c.start <= from && to <= c.end);
    return { text: text.slice(from - start, to - start), claim };
  });
}

function citationGroups(block: AnswerBlock): ActiveCitation[] {
  const bySource = new Map<number, number[]>();
  for (const c of block.citations) {
    const ids = bySource.get(c.source_index) ?? [];
    for (const id of c.sentence_ids) if (!ids.includes(id)) ids.push(id);
    bySource.set(c.source_index, ids);
  }
  return [...bySource.entries()].map(([sourceIndex, sentenceIds]) => ({ sourceIndex, sentenceIds }));
}

export function CitationChip({
  citation,
  active,
  onCite,
}: {
  citation: ActiveCitation;
  active: boolean;
  onCite: (c: ActiveCitation) => void;
}) {
  const n = citation.sourceIndex + 1;
  return (
    <button
      type="button"
      onClick={() => onCite(citation)}
      aria-label={`Show source ${n}`}
      aria-pressed={active}
      className={`mx-[1px] inline-flex h-[1.25rem] min-w-[1.25rem] translate-y-[-0.12em] items-center justify-center rounded-[4px] px-1 align-middle font-sans text-[0.7rem] font-semibold leading-none transition-colors ${
        active ? "bg-marker text-marker-ink" : "bg-teal-wash text-teal-deep hover:bg-marker-soft"
      }`}
    >
      {n}
    </button>
  );
}

export function AnswerText({ state, active, onCite }: Props) {
  const claims = state.confidence?.claims ?? [];
  const offsets = blockOffsets(state.blocks);
  const streaming = state.phase === "writing";
  return (
    <div className="font-serif text-[1.125rem] leading-[1.7] whitespace-pre-wrap text-ink">
      {state.blocks.map((block, i) => (
        <Fragment key={i}>
          {pieces(block.text, offsets[i]!.start, claims).map((p, j) =>
            p.claim && STATUS_CLASS[p.claim.status] ? (
              <span key={j} className={STATUS_CLASS[p.claim.status]} title={claimTitle(p.claim)}>
                {p.text}
              </span>
            ) : (
              <Fragment key={j}>{p.text}</Fragment>
            ),
          )}
          {citationGroups(block).map((c) => (
            <CitationChip
              key={c.sourceIndex}
              citation={c}
              onCite={onCite}
              active={active?.sourceIndex === c.sourceIndex && c.sentenceIds.some((id) => active.sentenceIds.includes(id))}
            />
          ))}
        </Fragment>
      ))}
      {streaming && <span className="caret" aria-hidden />}
    </div>
  );
}
