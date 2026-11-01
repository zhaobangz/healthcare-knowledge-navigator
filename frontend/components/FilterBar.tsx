"use client";

import type { Facets, Filters } from "@/lib/types";

const DOC_TYPE: Record<string, string> = {
  guideline: "Guidelines",
  protocol: "Treatment summaries",
  drug_label: "Drug labels",
  systematic_review: "Systematic reviews",
  rct: "Trials",
  review: "Reviews",
  other: "Other articles",
};
const SPECIALTY: Record<string, string> = {
  infectious_disease: "Infectious disease",
  hepatology: "Hepatology",
  pain_management: "Pain management",
  oncology: "Oncology",
};

function toggle(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

function Chip({ on, label, onClick }: { on: boolean; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={`rounded-full border px-2.5 py-1 text-[0.8rem] transition-colors ${
        on ? "border-teal bg-teal text-on-teal" : "border-rule bg-surface text-ink-soft hover:border-teal hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}

export function FilterBar({ facets, filters, onChange }: { facets: Facets | null; filters: Filters; onChange: (f: Filters) => void }) {
  if (!facets) return null;
  const active = filters.doc_types.length + filters.orgs.length + filters.specialties.length + (filters.year_min ? 1 : 0);
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="sr-only">Limit the search</legend>
      <div className="flex flex-wrap items-center gap-1.5">
        {facets.doc_types.map((t) => (
          <Chip key={t} on={filters.doc_types.includes(t)} label={DOC_TYPE[t] ?? t} onClick={() => onChange({ ...filters, doc_types: toggle(filters.doc_types, t) })} />
        ))}
        <span className="mx-1 h-4 w-px bg-rule" aria-hidden />
        {facets.orgs.map((o) => (
          <Chip key={o} on={filters.orgs.includes(o)} label={o} onClick={() => onChange({ ...filters, orgs: toggle(filters.orgs, o) })} />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {facets.specialties.map((s) => (
          <Chip key={s} on={filters.specialties.includes(s)} label={SPECIALTY[s] ?? s} onClick={() => onChange({ ...filters, specialties: toggle(filters.specialties, s) })} />
        ))}
        {facets.year_min && facets.year_max && (
          <label className="ml-1 inline-flex items-center gap-1.5 text-[0.8rem] text-ink-soft">
            Published since
            <select
              value={filters.year_min ?? ""}
              onChange={(e) => onChange({ ...filters, year_min: e.target.value ? Number(e.target.value) : null })}
              className="rounded-md border border-rule bg-surface px-1.5 py-0.5 text-ink"
            >
              <option value="">any year</option>
              {Array.from({ length: facets.year_max - facets.year_min + 1 }, (_, i) => facets.year_min! + i).map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>
        )}
        {active > 0 && (
          <button
            type="button"
            onClick={() => onChange({ ...filters, doc_types: [], orgs: [], specialties: [], year_min: null })}
            className="ml-1 text-[0.8rem] font-semibold text-teal hover:text-teal-deep"
          >
            Clear filters
          </button>
        )}
      </div>
    </fieldset>
  );
}
