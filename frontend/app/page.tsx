"use client";

import { AlertTriangle, ListTree, Loader2 } from "lucide-react";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { AnswerText, type ActiveCitation } from "@/components/AnswerView";
import { AskBox } from "@/components/AskBox";
import { ConfidencePanel, PreliminaryConfidence } from "@/components/ConfidencePanel";
import { FilterBar } from "@/components/FilterBar";
import { SourcePanel } from "@/components/SourcePanel";
import { TraceDrawer } from "@/components/TraceDrawer";
import { askStream, fetchFacets } from "@/lib/api";
import { initialState, isBusy, reducer, type AnswerState } from "@/lib/answerState";
import type { Facets, Filters } from "@/lib/types";

const EXAMPLES: { q: string; from: string }[] = [
  { q: "What is the recommended treatment for uncomplicated gonorrhea of the cervix, urethra, or rectum?", from: "CDC STI guidelines, 2021" },
  { q: "Who should be offered doxycycline postexposure prophylaxis, and at what dose?", from: "CDC doxy PEP guidelines, 2024" },
  { q: "At what age should average-risk adults start colorectal cancer screening?", from: "USPSTF, 2021" },
  { q: "How long should an initial opioid prescription for acute pain last?", from: "CDC opioid guideline, 2022" },
];

const EMPTY_FILTERS: Filters = { doc_types: [], orgs: [], specialties: [], year_min: null, year_max: null, exclude_superseded: false };

function StatusLine({ state }: { state: AnswerState }) {
  const text =
    state.phase === "retrieving"
      ? "Searching guidelines, drug labels and papers"
      : state.phase === "thinking"
        ? `Reading ${state.sources.length} source sections`
        : "Writing the answer with citations";
  return (
    <p className="flex items-center gap-2 text-sm text-ink-soft" role="status">
      <Loader2 size={15} className="animate-spin motion-reduce:animate-none" aria-hidden /> {text}
    </p>
  );
}

function Outcome({ state, onRetry }: { state: AnswerState; onRetry: () => void }) {
  if (state.phase === "insufficient")
    return (
      <div className="rounded-md border border-rule bg-surface p-5">
        <h2 className="font-semibold text-ink">Not enough evidence to answer</h2>
        <p className="mt-1 text-ink-soft">{state.reason}</p>
        <p className="mt-3 text-sm text-ink-soft">
          The knowledge base covers STIs, hepatitis B and C screening, opioid prescribing, and breast and colon cancer. Name the condition
          and population, or clear the filters, and ask again.
        </p>
      </div>
    );
  if (state.phase === "refused")
    return (
      <div className="rounded-md border border-rule bg-surface p-5">
        <h2 className="font-semibold text-ink">This question was declined</h2>
        <p className="mt-1 text-ink-soft">The model declined to answer, so no partial answer is shown. Rephrase the question in clinical terms.</p>
      </div>
    );
  if (state.phase === "error")
    return (
      <div className="flex items-start gap-3 rounded-md border border-bad/30 bg-bad-wash p-5">
        <AlertTriangle className="mt-0.5 shrink-0 text-bad" size={18} aria-hidden />
        <div>
          <h2 className="font-semibold text-ink">The answer could not be completed</h2>
          <p className="mt-1 text-ink-soft">{state.error}</p>
          <button type="button" onClick={onRetry} className="mt-3 text-sm font-semibold text-teal hover:text-teal-deep">
            Ask again
          </button>
        </div>
      </div>
    );
  return null;
}

export default function Home() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [facetsError, setFacetsError] = useState(false);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [active, setActive] = useState<ActiveCitation | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [askKey, setAskKey] = useState(0);
  const [draft, setDraft] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchFacets(ctrl.signal)
      .then(setFacets)
      .catch((e: unknown) => {
        if ((e as Error).name !== "AbortError") setFacetsError(true);
      });
    return () => ctrl.abort();
  }, []);

  const ask = useCallback(
    async (question: string) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setActive(null);
      dispatch({ type: "start", question });
      try {
        for await (const event of askStream(question, filters, ctrl.signal)) dispatch({ type: "event", event });
      } catch (e) {
        if ((e as Error).name === "AbortError") dispatch({ type: "abort" });
        else dispatch({ type: "event", event: { event: "error", data: { kind: "network", message: "The answer service is unreachable. Is the backend running on port 8002?" } } });
      }
    },
    [filters],
  );

  const busy = isBusy(state.phase);
  const hasAnswer = state.blocks.length > 0;
  const startExample = (q: string) => {
    setDraft(q);
    setAskKey((k) => k + 1);
    void ask(q);
  };

  return (
    <div className="min-h-dvh">
      <p className="border-b border-rule bg-surface px-4 py-2 text-center text-[0.8rem] text-ink-soft">
        For health professionals and education. Not medical advice: answers come only from the listed sources and can be wrong. Check the
        original before acting.
      </p>

      <header className="mx-auto flex max-w-[78rem] flex-wrap items-end justify-between gap-3 px-4 pt-8 pb-5 sm:px-6">
        <div>
          <h1 className="font-serif text-[1.9rem] leading-tight font-semibold tracking-[-0.01em] text-ink">Healthcare Knowledge Navigator</h1>
          <p className="mt-1 max-w-[62ch] text-ink-soft">
            {facets
              ? `Answers from ${facets.documents} clinical documents (${facets.orgs.join(", ")}), with every statement cited to the sentence it came from.`
              : facetsError
                ? "The knowledge base is unreachable. Start the backend with make dev."
                : "Answers from clinical guidelines, drug labels and open-access research, with every statement cited."}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setTraceOpen(true)}
          disabled={!state.trace}
          className="inline-flex items-center gap-1.5 rounded-md border border-rule bg-surface px-3 py-1.5 text-sm font-semibold text-ink hover:border-teal disabled:opacity-40"
        >
          <ListTree size={15} aria-hidden /> Retrieval trace
        </button>
      </header>

      <main className="mx-auto grid max-w-[78rem] gap-8 px-4 pb-16 sm:px-6 lg:grid-cols-[minmax(0,1fr)_25rem]">
        <div className="flex min-w-0 flex-col gap-5">
          <AskBox key={askKey} initial={draft} busy={busy} onAsk={(q) => void ask(q)} onStop={() => abortRef.current?.abort()} />
          <FilterBar facets={facets} filters={filters} onChange={setFilters} />

          {state.phase === "idle" && (
            <section aria-label="Example questions" className="mt-2">
              <h2 className="text-sm font-semibold text-ink-soft">Try a question the sources cover</h2>
              <ul className="mt-2 divide-y divide-rule rounded-md border border-rule bg-surface">
                {EXAMPLES.map((ex) => (
                  <li key={ex.q}>
                    <button type="button" onClick={() => startExample(ex.q)} className="w-full px-4 py-3 text-left hover:bg-teal-wash/50">
                      <span className="block font-serif text-[1.02rem] text-ink">{ex.q}</span>
                      <span className="text-[0.8rem] text-ink-faint">{ex.from}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {state.phase !== "idle" && (
            <article aria-live="polite" aria-busy={busy} className="flex flex-col gap-4">
              <h2 className="font-serif text-[1.3rem] leading-snug text-ink">{state.question}</h2>
              {busy && <StatusLine state={state} />}
              {state.notice && <p className="rounded-md bg-teal-wash px-4 py-2.5 text-sm text-teal-deep">{state.notice}</p>}
              {hasAnswer && <AnswerText state={state} active={active} onCite={setActive} />}
              {state.servedBy && (
                <p className="text-[0.8rem] text-ink-faint">
                  Completed by {state.servedBy.to_model} after {state.servedBy.from_model} handed over.
                </p>
              )}
              <Outcome state={state} onRetry={() => void ask(state.question)} />
              {busy && state.preliminary?.label && state.preliminary.label !== "unscored" && <PreliminaryConfidence label={state.preliminary.label} />}
              {state.phase === "answered" && state.confidence && <ConfidencePanel confidence={state.confidence} />}
            </article>
          )}
        </div>

        <aside className="min-w-0 lg:sticky lg:top-6 lg:max-h-[calc(100dvh-3rem)] lg:self-start lg:overflow-y-auto">
          <SourcePanel sources={state.sources} blocks={state.blocks} active={active} onSelect={setActive} />
        </aside>
      </main>

      <TraceDrawer trace={state.trace} open={traceOpen} onClose={() => setTraceOpen(false)} />
    </div>
  );
}
