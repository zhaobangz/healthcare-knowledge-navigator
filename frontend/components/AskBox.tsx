"use client";

import { Search, Square } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";

interface Props {
  busy: boolean;
  onAsk: (question: string) => void;
  onStop: () => void;
  initial?: string;
}

export function AskBox({ busy, onAsk, onStop, initial = "" }: Props) {
  const [value, setValue] = useState(initial);
  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const q = value.trim();
    if (q.length >= 3 && !busy) onAsk(q);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };
  return (
    <form onSubmit={submit} className="rounded-lg border border-rule bg-surface p-2 shadow-[0_1px_0_var(--color-rule)] focus-within:border-teal">
      <label htmlFor="question" className="sr-only">
        Clinical question
      </label>
      <textarea
        id="question"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        rows={2}
        maxLength={1000}
        placeholder="Ask a clinical question, for example: first-line treatment for uncomplicated gonorrhea in adults"
        className="block w-full resize-none bg-transparent px-2 py-1.5 text-[1.05rem] leading-relaxed text-ink outline-none placeholder:text-ink-faint"
      />
      <div className="flex items-center justify-between gap-2 px-2 pb-1">
        <span className="text-[0.75rem] text-ink-faint">Enter to ask, Shift+Enter for a new line</span>
        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className="inline-flex items-center gap-1.5 rounded-md border border-rule px-3 py-1.5 text-sm font-semibold text-ink hover:bg-paper"
          >
            <Square size={14} aria-hidden /> Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={value.trim().length < 3}
            className="inline-flex items-center gap-1.5 rounded-md bg-teal px-3.5 py-1.5 text-sm font-semibold text-on-teal hover:bg-teal-deep disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Search size={14} aria-hidden /> Ask
          </button>
        )}
      </div>
    </form>
  );
}
