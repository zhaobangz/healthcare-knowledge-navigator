"use client";

import { X } from "lucide-react";
import { useEffect, useRef } from "react";

import type { Trace } from "@/lib/types";

const ARM_LABEL: Record<string, string> = {
  "dense:original": "Dense, question",
  "dense:rewritten": "Dense, rewritten",
  "dense:hyde": "Dense, HyDE",
  "fts:keywords": "Keywords",
  "fts:original": "Keywords, question",
};

export function TraceDrawer({ trace, open, onClose }: { trace: Trace | null; open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const plan = trace?.plan;
  const retrieval = trace?.retrieval;
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      className="m-0 ml-auto h-dvh max-h-dvh w-full max-w-[44rem] bg-surface p-0 text-ink shadow-xl backdrop:bg-ink/30"
      aria-label="Retrieval trace"
    >
      <div className="sticky top-0 flex items-center justify-between border-b border-rule bg-surface px-5 py-3">
        <h2 className="text-base font-semibold">How this answer was retrieved</h2>
        <button type="button" onClick={onClose} className="rounded p-1 text-ink-soft hover:bg-paper" aria-label="Close">
          <X size={18} />
        </button>
      </div>
      {!trace ? (
        <p className="p-5 text-sm text-ink-soft">Ask a question to see its retrieval trace.</p>
      ) : (
        <div className="flex flex-col gap-6 p-5 text-sm">
          <section>
            <h3 className="font-semibold">Query plan</h3>
            {plan ? (
              <dl className="mt-2 grid grid-cols-[9rem_1fr] gap-x-3 gap-y-2">
                <dt className="text-ink-soft">Rewritten</dt>
                <dd>{plan.rewritten_query}</dd>
                <dt className="text-ink-soft">Keywords</dt>
                <dd className="flex flex-wrap gap-1">
                  {plan.keywords.map((k) => (
                    <span key={k} className="rounded bg-paper px-1.5 py-0.5 text-[0.8rem]">
                      {k}
                    </span>
                  ))}
                </dd>
                {plan.sub_questions.length > 0 && (
                  <>
                    <dt className="text-ink-soft">Sub-questions</dt>
                    <dd>
                      <ul className="list-disc pl-4">
                        {plan.sub_questions.map((q) => (
                          <li key={q}>{q}</li>
                        ))}
                      </ul>
                    </dd>
                  </>
                )}
                <dt className="text-ink-soft">Inferred filters</dt>
                <dd>
                  {[...plan.soft_filters.doc_types, ...plan.soft_filters.specialties].join(", ") || "none"}
                  {retrieval?.soft_filters_dropped ? " (dropped: too few matches)" : ""}
                </dd>
                <dt className="text-ink-soft">HyDE probe</dt>
                <dd className="font-serif text-ink-soft italic">{plan.hyde_passage}</dd>
              </dl>
            ) : (
              <p className="mt-2 text-ink-soft">Not used{trace.plan_error ? `: ${trace.plan_error}` : ""}. Retrieval ran on the question as written.</p>
            )}
          </section>
          {retrieval && (
            <section>
              <h3 className="font-semibold">Ranked passages</h3>
              <p className="mt-1 text-ink-soft">
                {retrieval.pool_size} candidates fused with reciprocal rank fusion, then reranked by a cross-encoder.{" "}
                {Object.entries(retrieval.timings_ms)
                  .map(([k, v]) => `${k} ${Math.round(v)} ms`)
                  .join(", ")}
                .
              </p>
              <ol className="mt-3 flex flex-col divide-y divide-rule rounded-md border border-rule">
                {retrieval.candidates.map((c, i) => (
                  <li key={c.chunk_id} className="grid grid-cols-[1.5rem_1fr] gap-2 p-3">
                    <span className="font-semibold text-ink-soft tabular-nums">{i + 1}</span>
                    <div className="min-w-0">
                      <p className="truncate font-semibold" title={c.doc}>
                        {c.doc}
                      </p>
                      <p className="truncate text-[0.8rem] text-ink-faint" title={c.heading}>
                        {c.heading}
                      </p>
                      <p className="mt-1 line-clamp-2 text-[0.8rem] text-ink-soft">{c.preview}</p>
                      <p className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[0.75rem] text-ink-soft tabular-nums">
                        <span>Rerank {c.rerank?.toFixed(2) ?? "n/a"}</span>
                        <span>RRF {c.rrf.toFixed(4)}</span>
                        {Object.entries(c.arm_ranks).map(([arm, rank]) => (
                          <span key={arm}>
                            {ARM_LABEL[arm] ?? arm} #{rank}
                          </span>
                        ))}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}
    </dialog>
  );
}
