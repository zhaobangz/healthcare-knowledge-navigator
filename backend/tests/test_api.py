from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hkn.api import main as api_main
from hkn.api.sse import format_event


def parse_sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for frame in body.strip().split("\n\n"):
        lines = frame.split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


def test_format_event_is_one_sse_frame() -> None:
    frame = format_event("text", {"block": 0, "text": "≥150 kg\nnext"})
    assert frame.startswith("event: text\ndata: ")
    assert frame.endswith("\n\n")
    assert frame.count("\n\n") == 1  # embedded newlines are JSON-escaped, never raw


def test_ask_streams_events_in_order(monkeypatch) -> None:
    async def fake_events(question, filters):
        assert question == "First-line gonorrhea treatment?"
        assert filters.doc_types == ["guideline"]
        yield {"event": "sources", "data": {"sources": []}}
        yield {"event": "text", "data": {"block": 0, "text": "Ceftriaxone"}}
        yield {"event": "done", "data": {"status": "answered"}}

    monkeypatch.setattr(api_main, "ask_events_async", fake_events)
    client = TestClient(api_main.app)
    res = client.post(
        "/ask",
        json={
            "question": "  First-line gonorrhea treatment?  ",
            "filters": {"doc_types": ["guideline"]},
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["x-accel-buffering"] == "no"
    assert [name for name, _ in parse_sse(res.text)] == ["sources", "text", "done"]


def test_ask_rejects_empty_question() -> None:
    client = TestClient(api_main.app)
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_pipeline_exception_becomes_error_event(monkeypatch) -> None:
    async def broken(question, filters):
        yield {"event": "sources", "data": {"sources": []}}
        raise RuntimeError("boom")

    monkeypatch.setattr(api_main, "ask_events_async", broken)
    res = TestClient(api_main.app).post("/ask", json={"question": "anything clinical?"})
    names = [name for name, _ in parse_sse(res.text)]
    assert names == ["sources", "error", "done"]
