"use client";

import { ExternalLink } from "lucide-react";
import { useEffect, useRef } from "react";

import type { ActiveCitation } from "@/components/AnswerView";
import type { AnswerBlock, Source } from "@/lib/types";

const DOC_TYPE: Record<string, string> = {
  guideline: "Clinical guideline",
  protocol: "Treatment summary",
  systematic_review: "Systematic review",
  rct: "Randomized trial",
  review: "Review article",
  drug_label: "FDA drug label",
  other: "Research article",
};

interface Props {
  sources: Source[];
  blocks: AnswerBlock[];
  active: ActiveCitation | null;
  onSelect: (c: ActiveCitation) => void;
}

function textFragmentUrl(url: string, sentence: string | undefined): string {
  if (!sentence) return url;
  const snippet = encodeURIComponent(sentence.split(/\s+/).slice(0, 8).join(" "));
  return url.includes("#") ? `${url}:~:text=${snippet}` : `${url}#:~:text=${snippet}`;
}

export function SourcePanel({ sources, blocks, active, onSelect }: Props) {
  const cited = new Map<number, Set<number>>();
  for (const b of blocks)
    for (const c of b.citations) {
      const set = cited.get(c.source_index) ?? new Set<number>();
      c.sentence_ids.forEach((id) => set.add(id));
      cited.set(c.source_index, set);
    }
  const markRef = useRef<HTMLElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    // Scroll only the section's own box, never the page, to bring the mark into view.
    const box = scrollRef.current;
    const mark = markRef.current;
    if (!box || !mark) return;
    const top = mark.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
    box.scrollTo({ top: Math.max(0, top - 24), behavior: "smooth" });
  }, [active]);

  if (!sources.length) return null;
  const citedSources = sources.filter((s) => cited.has(s.index));
  const shown = citedSources.length ? citedSources : sources;
  const expandedIndex = active?.sourceIndex ?? shown[0]?.index;

  return (
    <section aria-label="Sources" className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-ink-soft">
        {citedSources.length ? `Cited sources (${citedSources.length} of ${sources.length} retrieved)` : `Retrieved sources (${sources.length})`}
      </h2>
      <ol className="flex flex-col gap-2">
        {shown.map((src) => {
          const expanded = src.index === expandedIndex;
          const citedIds = cited.get(src.index) ?? new Set<number>();
          const activeIds = active?.sourceIndex === src.index ? new Set(active.sentenceIds) : new Set<number>();
          const firstActive = src.sentences.find((s) => activeIds.has(s.id));
          return (
            <li key={src.index} className="rounded-md border border-rule bg-surface">
              <button
                type="button"
                onClick={() => onSelect({ sourceIndex: src.index, sentenceIds: [...citedIds] })}
                aria-expanded={expanded}
                className="flex w-full items-start gap-3 rounded-md p-3 text-left hover:bg-teal-wash/40"
              >
                <span
                  className={`mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-[4px] px-1 text-xs font-semibold ${
                    active?.sourceIndex === src.index ? "bg-marker text-marker-ink" : "bg-teal-wash text-teal-deep"
                  }`}
                >
                  {src.number}
                </span>
                <span className="min-w-0">
                  <span className="block text-[0.95rem] font-semibold leading-snug text-ink">{src.doc_title}</span>
                  <span className="block text-[0.8rem] text-ink-soft">
                    {src.org}, {DOC_TYPE[src.doc_type] ?? src.doc_type}
                    {src.pub_year ? `, ${src.pub_year}` : ""}
                    {src.superseded ? " (superseded edition)" : ""}
                  </span>
                  {src.heading_path.length > 0 && (
                    <span className="mt-1 block truncate text-[0.8rem] text-ink-faint" title={src.heading_path.join(" › ")}>
                      {src.heading_path.join(" › ")}
                    </span>
                  )}
                </span>
              </button>
              {expanded && (
                <div className="border-t border-rule px-4 pt-3 pb-4">
                  <div ref={scrollRef} className="max-h-[22rem] overflow-y-auto pr-1 font-serif text-[0.95rem] leading-[1.65] text-ink-soft">
                    {src.sentences.map((s) => {
                      const isActive = activeIds.has(s.id);
                      const isCited = citedIds.has(s.id);
                      const body = isActive ? (
                        <mark ref={s.id === firstActive?.id ? markRef : undefined} className="marker">
                          {s.text}
                        </mark>
                      ) : (
                        <span className={isCited ? "text-ink underline decoration-marker decoration-2 underline-offset-4" : undefined}>
                          {s.text}
                        </span>
                      );
                      return s.kind === "sentence" ? (
                        <span key={s.id}>{body} </span>
                      ) : (
                        <span key={s.id} className={`my-1 block ${s.kind === "caption" ? "font-sans text-[0.8rem] font-semibold text-ink" : ""}`}>
                          {s.kind === "list_item" ? "• " : ""}
                          {body}
                        </span>
                      );
                    })}
                  </div>
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[0.75rem] text-ink-faint">
                    <span>{src.license}{src.authors ? `. ${src.authors}` : ""}</span>
                    <a
                      href={textFragmentUrl(src.url, firstActive?.text)}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 font-semibold text-teal hover:text-teal-deep"
                    >
                      Open original <ExternalLink size={13} aria-hidden />
                    </a>
                  </div>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
