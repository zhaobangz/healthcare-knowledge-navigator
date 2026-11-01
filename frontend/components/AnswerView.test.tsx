import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AnswerText } from "./AnswerView";
import { initialState, type AnswerState } from "@/lib/answerState";
import type { Claim } from "@/lib/types";

const claim = (start: number, end: number, status: Claim["status"], extra: Partial<Claim> = {}): Claim => ({
  start,
  end,
  text: "",
  cited: status !== "uncited",
  substantive: true,
  source_indices: [0],
  entailment: 0.9,
  contradiction: 0,
  facts_checked: 1,
  facts_missing: [],
  status,
  ...extra,
});

describe("AnswerText", () => {
  const citation = { source_index: 0, section_id: 1, document_id: 1, sentence_ids: [7], cited_text: "", title: "", url: "" };
  const text1 = "Give ceftriaxone 500 mg IM.";
  const text2 = " Give 250 mg if heavy.";
  const state: AnswerState = {
    ...initialState,
    phase: "answered",
    blocks: [
      { text: text1, citations: [citation] },
      { text: text2, citations: [{ ...citation, sentence_ids: [8] }] },
    ],
    confidence: {
      reliability: { score: 0.4, band: "Low", calibrated: false, features: { nli_min: 0, facts_ok: 0.5, retrieval: 0.9, coverage: 1, relevance: 0.9 }, reasons: [], bands: { high: 0.8, moderate: 0.55 } },
      evidence: { rating: "Strong", score: 1, n_documents: 1, reasons: [] },
      claims: [claim(0, text1.length, "supported"), claim(text1.length + 1, text1.length + text2.length, "conflict", { facts_missing: ["250"] })],
    },
  };

  it("renders a citation chip per cited block and marks only the conflicting claim", () => {
    const onCite = vi.fn();
    const { container } = render(<AnswerText state={state} active={null} onCite={onCite} />);
    const chips = screen.getAllByRole("button", { name: "Show source 1" });
    expect(chips).toHaveLength(2);
    const marked = container.querySelectorAll(".claim-conflict");
    expect(marked).toHaveLength(1);
    expect(marked[0]!.textContent).toBe("Give 250 mg if heavy.");
    expect(marked[0]!.getAttribute("title")).toContain("250");
    fireEvent.click(chips[1]!);
    expect(onCite).toHaveBeenCalledWith({ sourceIndex: 0, sentenceIds: [8] });
  });
});
