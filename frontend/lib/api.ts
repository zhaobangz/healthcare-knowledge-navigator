import { SseParser } from "./sse";
import type { Facets, Filters, StreamEvent } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

export async function fetchFacets(signal?: AbortSignal): Promise<Facets> {
  const res = await fetch(`${API_URL}/facets`, { signal });
  if (!res.ok) throw new Error(`facets request failed (${res.status})`);
  return res.json();
}

/** POST a question and yield typed events as they stream in. Aborting the signal closes
 * the connection, which the backend treats as "stop generating". */
export async function* askStream(
  question: string,
  filters: Filters,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, filters }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = res.status === 422 ? "Questions must be between 3 and 1,000 characters." : `The server answered ${res.status}.`;
    yield { event: "error", data: { kind: "http", message: detail, status: res.status } };
    yield { event: "done", data: { status: "error" } };
    return;
  }
  const reader = res.body.getReader();
  const parser = new SseParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const raw of parser.push(value)) {
        yield { event: raw.event, data: JSON.parse(raw.data) } as StreamEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
