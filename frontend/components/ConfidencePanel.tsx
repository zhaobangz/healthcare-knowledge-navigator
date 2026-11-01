"use client";

import type { Claim, Confidence } from "@/lib/types";

const FEATURE_LABELS: Record<string, string> = {
  nli_min: "Cited passages entail each statement",
  facts_ok: "Numbers and drug names match the sources",
  retrieval: "Best passage matches the question",
  coverage: "Statements that carry a citation",
  relevance: "Answer addresses the question",
};

const BAND_COLOR: Record<string, string> = { High: "var(--color-ok)", Moderate: "var(--color-warn)", Low: "var(--color-bad)" };

function claimSummary(claims: Claim[]): string {
  const substantive = claims.filter((c) => c.substantive);
  const supported = substantive.filter((c) => c.status === "supported").length;
  return `${supported} of ${substantive.length} statements supported by their citations`;
}

export function PreliminaryConfidence({ label }: { label: string }) {
  return (
    <p className="text-sm text-ink-soft">
      Retrieval match: <span className="font-semibold text-ink">{label}</span>. Reliability is scored when the answer is complete.
    </p>
  );
}

export function ConfidencePanel({ confidence }: { confidence: Confidence }) {
  const { reliability, evidence, claims } = confidence;
  const thresholds = reliability.bands;
  const pct = Math.round(reliability.score * 100);
  const color = BAND_COLOR[reliability.band] ?? "var(--color-ink)";
  return (
    <section aria-label="Confidence" className="grid gap-4 sm:grid-cols-[1.4fr_1fr]">
      <div className="rounded-md border border-rule bg-surface p-4">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-sm font-semibold text-ink-soft">Answer reliability</h2>
          <span className="text-2xl font-semibold tabular-nums" style={{ color }}>
            {reliability.band} <span className="text-base text-ink-soft">{pct}%</span>
          </span>
        </div>
        <div
          className="relative mt-3 h-2 rounded-full bg-paper"
          role="meter"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label="Answer reliability"
        >
          <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${pct}%`, background: color }} />
          {[thresholds.moderate, thresholds.high].map((t) => (
            <span key={t} className="absolute -top-1 h-4 w-px bg-ink-faint" style={{ left: `${t * 100}%` }} aria-hidden />
          ))}
        </div>
        <p className="mt-2 text-[0.8rem] text-ink-faint">
          {reliability.calibrated
            ? "Calibrated probability that the answer is correct and supported, fitted on labeled answers."
            : "Uncalibrated: prior weights, not yet fitted on labeled answers. Treat as a ranking, not a probability."}
        </p>
        <p className="mt-3 text-sm text-ink">{claimSummary(claims)}.</p>
        <dl className="mt-3 grid gap-1.5">
          {Object.entries(reliability.features).map(([name, value]) => (
            <div key={name} className="grid grid-cols-[1fr_auto] items-center gap-3 text-[0.8rem]">
              <dt className="text-ink-soft">{FEATURE_LABELS[name] ?? name}</dt>
              <dd className="tabular-nums text-ink">{Math.round(value * 100)}%</dd>
            </div>
          ))}
        </dl>
        {reliability.reasons.length > 0 && (
          <ul className="mt-3 list-disc pl-5 text-[0.8rem] text-ink-soft">
            {reliability.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>
      <div className="rounded-md border border-rule bg-surface p-4">
        <h2 className="text-sm font-semibold text-ink-soft">Evidence strength</h2>
        <p className="mt-1 text-2xl font-semibold text-ink">{evidence.rating}</p>
        <p className="mt-1 text-[0.8rem] text-ink-faint">How authoritative and current the cited sources are. Rated separately from reliability.</p>
        <ul className="mt-3 flex flex-col gap-1 text-[0.85rem] text-ink-soft">
          {evidence.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
