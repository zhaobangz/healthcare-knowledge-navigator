import type { AnswerBlock, Confidence, Source, StreamEvent, Trace } from "./types";

export type Phase = "idle" | "retrieving" | "thinking" | "writing" | "answered" | "insufficient" | "refused" | "error";

export interface AnswerState {
  phase: Phase;
  question: string;
  trace: Trace | null;
  sources: Source[];
  preliminary: { retrieval: number | null; label: string } | null;
  notice: string | null;
  blocks: AnswerBlock[];
  servedBy: { from_model: string; to_model: string } | null;
  confidence: Confidence | null;
  reason: string | null;
  error: string | null;
}

export const initialState: AnswerState = {
  phase: "idle",
  question: "",
  trace: null,
  sources: [],
  preliminary: null,
  notice: null,
  blocks: [],
  servedBy: null,
  confidence: null,
  reason: null,
  error: null,
};

export type Action = { type: "start"; question: string } | { type: "event"; event: StreamEvent } | { type: "abort" };

export function reducer(state: AnswerState, action: Action): AnswerState {
  if (action.type === "start") return { ...initialState, phase: "retrieving", question: action.question };
  if (action.type === "abort") {
    return state.phase === "answered" ? state : { ...state, phase: "error", error: "Stopped before the answer finished." };
  }
  const ev = action.event;
  switch (ev.event) {
    case "trace":
      return { ...state, trace: ev.data };
    case "sources":
      return { ...state, sources: ev.data.sources };
    case "confidence_preliminary":
      return { ...state, preliminary: ev.data };
    case "notice":
      return { ...state, notice: ev.data.text };
    case "status":
      return { ...state, phase: ev.data.phase };
    case "block_start": {
      const blocks = state.blocks.slice();
      while (blocks.length <= ev.data.block) blocks.push({ text: "", citations: [] });
      return { ...state, blocks };
    }
    case "text":
      return { ...state, blocks: updateBlock(state.blocks, ev.data.block, (b) => ({ ...b, text: b.text + ev.data.text })) };
    case "citation": {
      const { block, ...citation } = ev.data;
      return { ...state, blocks: updateBlock(state.blocks, block, (b) => ({ ...b, citations: [...b.citations, citation] })) };
    }
    case "served_by":
      return { ...state, servedBy: ev.data };
    case "confidence":
      return { ...state, confidence: ev.data };
    case "insufficient":
      return { ...state, phase: "insufficient", reason: ev.data.reason, blocks: [] };
    case "refusal":
      // The whole fallback chain declined: partial text must not stay on screen.
      return { ...state, phase: "refused", blocks: [], confidence: null };
    case "error":
      return { ...state, phase: "error", error: ev.data.message };
    case "done":
      return ev.data.status === "answered" ? { ...state, phase: "answered" } : state;
    default:
      return state;
  }
}

function updateBlock(blocks: AnswerBlock[], index: number, fn: (b: AnswerBlock) => AnswerBlock): AnswerBlock[] {
  const next = blocks.slice();
  while (next.length <= index) next.push({ text: "", citations: [] });
  next[index] = fn(next[index]!);
  return next;
}

/** Character ranges of each block within the concatenated answer text. */
export function blockOffsets(blocks: AnswerBlock[]): { start: number; end: number }[] {
  let pos = 0;
  return blocks.map((b) => {
    const range = { start: pos, end: pos + b.text.length };
    pos += b.text.length;
    return range;
  });
}

export const isBusy = (phase: Phase) => phase === "retrieving" || phase === "thinking" || phase === "writing";
