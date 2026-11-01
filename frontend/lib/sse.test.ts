import { describe, expect, it } from "vitest";

import { reducer, initialState, blockOffsets } from "./answerState";
import { SseParser } from "./sse";
import type { StreamEvent } from "./types";

describe("SseParser", () => {
  it("reassembles frames split across chunks, including inside a UTF-8 character", () => {
    const parser = new SseParser();
    const bytes = new TextEncoder().encode('event: text\ndata: {"block":0,"text":"≥150 kg"}\n\nevent: done\ndata: {}\n\n');
    const split = 30; // lands inside the 3-byte "≥"
    const first = parser.push(bytes.slice(0, split));
    const second = parser.push(bytes.slice(split));
    const all = [...first, ...second];
    expect(all.map((e) => e.event)).toEqual(["text", "done"]);
    expect(JSON.parse(all[0]!.data).text).toBe("≥150 kg");
  });

  it("ignores comments and joins multi-line data", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\n\nevent: a\ndata: 1\ndata: 2\n\n")).toEqual([{ event: "a", data: "1\n2" }]);
  });
});

describe("answer reducer", () => {
  const run = (events: StreamEvent[]) =>
    events.reduce((s, event) => reducer(s, { type: "event", event }), reducer(initialState, { type: "start", question: "q" }));

  it("builds block-scoped text and citations in arrival order", () => {
    const state = run([
      { event: "status", data: { phase: "writing" } },
      { event: "block_start", data: { block: 0 } },
      { event: "text", data: { block: 0, text: "CDC recommends " } },
      { event: "block_start", data: { block: 1 } },
      { event: "text", data: { block: 1, text: "ceftriaxone 500 mg" } },
      {
        event: "citation",
        data: { block: 1, source_index: 0, section_id: 1, document_id: 1, sentence_ids: [7], cited_text: "", title: "", url: "" },
      },
      { event: "done", data: { status: "answered" } },
    ]);
    expect(state.phase).toBe("answered");
    expect(state.blocks.map((b) => b.text).join("")).toBe("CDC recommends ceftriaxone 500 mg");
    expect(state.blocks[1]!.citations[0]!.sentence_ids).toEqual([7]);
    expect(blockOffsets(state.blocks)).toEqual([
      { start: 0, end: 15 },
      { start: 15, end: 33 },
    ]);
  });

  it("discards partial text when the model declines", () => {
    const state = run([
      { event: "block_start", data: { block: 0 } },
      { event: "text", data: { block: 0, text: "Partial" } },
      { event: "refusal", data: { category: "bio" } },
    ]);
    expect(state.phase).toBe("refused");
    expect(state.blocks).toEqual([]);
  });
});
