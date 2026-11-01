"""Server-Sent Events framing.

POST + `text/event-stream` (the browser reads it with fetch + ReadableStream, since
EventSource cannot POST). Buffering is disabled end to end: `X-Accel-Buffering: no` for
nginx-style proxies, `Cache-Control: no-cache, no-transform` for everything else.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from starlette.requests import Request

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def format_event(name: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"


async def relay(request: Request, events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    """Frame events; stop (and so cancel upstream generation) when the client leaves.

    Breaking out of the `async for` closes the `events` generator, which exits the
    Anthropic stream's `async with` and closes that HTTP connection — a closed browser
    tab stops paying for tokens.
    """
    try:
        async for ev in events:
            if await request.is_disconnected():
                break
            yield format_event(ev["event"], ev["data"])
    except Exception as exc:  # never leave the client hanging on a half-open stream
        yield format_event("error", {"kind": "internal", "message": type(exc).__name__})
        yield format_event("done", {"status": "error"})
    finally:
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            await aclose()
