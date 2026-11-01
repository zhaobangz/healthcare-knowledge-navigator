// Mirrors the event payloads emitted by backend/src/hkn/ask.py and generation/answer.py.

export type DocType = "guideline" | "protocol" | "systematic_review" | "rct" | "review" | "drug_label" | "other";

export interface SourceSentence {
  id: number;
  text: string;
  kind: "sentence" | "list_item" | "table_row" | "caption";
  matched: boolean;
}

export interface Source {
  index: number;
  number: number;
  title: string;
  doc_title: string;
  org: string;
  doc_type: DocType;
  pub_year: number | null;
  url: string;
  license: string;
  authors: string;
  superseded: boolean;
  heading_path: string[];
  rerank: number | null;
  sentences: SourceSentence[];
}

export interface Citation {
  source_index: number;
  section_id: number;
  document_id: number;
  sentence_ids: number[];
  cited_text: string;
  title: string;
  url: string;
}

export interface AnswerBlock {
  text: string;
  citations: Citation[];
}

export type ClaimStatus = "supported" | "weak" | "conflict" | "uncited" | "connective";

export interface Claim {
  start: number;
  end: number;
  text: string;
  cited: boolean;
  substantive: boolean;
  source_indices: number[];
  entailment: number | null;
  contradiction: number | null;
  facts_checked: number;
  facts_missing: string[];
  status: ClaimStatus;
}

export interface Reliability {
  score: number;
  band: "High" | "Moderate" | "Low";
  calibrated: boolean;
  features: Record<"nli_min" | "facts_ok" | "retrieval" | "coverage" | "relevance", number>;
  reasons: string[];
  bands: { high: number; moderate: number };
}

export interface Evidence {
  rating: "Strong" | "Moderate" | "Limited";
  score: number;
  n_documents: number;
  reasons: string[];
}

export interface Confidence {
  reliability: Reliability;
  evidence: Evidence;
  claims: Claim[];
}

export interface TraceCandidate {
  chunk_id: number;
  doc: string;
  heading: string;
  rrf: number;
  rerank: number | null;
  arm_ranks: Record<string, number>;
  preview: string;
}

export interface QueryPlan {
  is_clinical: boolean;
  asks_personal_advice: boolean;
  rewritten_query: string;
  keywords: string[];
  sub_questions: string[];
  soft_filters: { doc_types: string[]; specialties: string[]; year_min: number | null };
  hyde_passage: string;
}

export interface Trace {
  plan: QueryPlan | null;
  plan_error: string | null;
  retrieval: {
    queries: Record<string, string>;
    keywords: string[];
    soft_filters: Record<string, unknown> | null;
    soft_filters_dropped: boolean;
    pool_size: number;
    timings_ms: Record<string, number>;
    candidates: TraceCandidate[];
  } | null;
}

export interface Filters {
  doc_types: string[];
  orgs: string[];
  specialties: string[];
  year_min: number | null;
  year_max: number | null;
  exclude_superseded: boolean;
}

export interface Facets {
  doc_types: string[];
  orgs: string[];
  specialties: string[];
  year_min: number | null;
  year_max: number | null;
  documents: number;
  chunks: number;
}

export type StreamEvent =
  | { event: "trace"; data: Trace }
  | { event: "sources"; data: { sources: Source[] } }
  | { event: "confidence_preliminary"; data: { retrieval: number | null; label: string } }
  | { event: "notice"; data: { kind: string; text: string } }
  | { event: "status"; data: { phase: "thinking" | "writing" } }
  | { event: "block_start"; data: { block: number } }
  | { event: "text"; data: { block: number; text: string } }
  | ({ event: "citation"; data: { block: number } & Citation })
  | { event: "block_stop"; data: { block: number } }
  | { event: "served_by"; data: { from_model: string; to_model: string } }
  | { event: "confidence"; data: Confidence }
  | { event: "insufficient"; data: { stage: string; reason: string } }
  | { event: "refusal"; data: { category: string | null } }
  | { event: "error"; data: { kind: string; message: string; status?: number } }
  | { event: "done"; data: { status: string } };
