"""Cited answer generation (concept 14).

Retrieved sections go to Claude as top-level `search_result` content blocks with
citations enabled, one text block per stored sentence, so every citation in the reply
resolves to exact sentences. Structured outputs are deliberately NOT used here: they are
incompatible with citations (the API rejects the combination).

Streaming is translated into block-scoped `AnswerEvent`s by one pure state machine
(`StreamTranslator`) shared by the sync CLI path and the async API path. Citations arrive
*inside* a text block, interleaved with its text deltas, so events carry the block index.
A server-side fallback inserts a `fallback` content block mid-stream; it becomes a
`served_by` event. A final `refusal` stop reason means the whole chain declined: the
consumer must discard any partial text it rendered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic

from hkn.config import Settings, get_settings
from hkn.generation.citations import CitationMapper, MappedCitation
from hkn.generation.prompts import ANSWER_SYSTEM
from hkn.llm import NO_CREDENTIALS_MESSAGE, get_async_client, get_client, is_missing_credentials
from hkn.retrieval.expand import SourceSection

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnswerEvent:
    type: str  # status | block_start | text | citation | block_stop | served_by | refusal | done | error
    data: dict[str, Any] = field(default_factory=dict)


def build_request(
    question: str, sources: list[SourceSection], settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    content: list[dict[str, Any]] = [
        {
            "type": "search_result",
            "source": src.source_url,
            "title": src.title,
            "content": [{"type": "text", "text": s.text} for s in src.sentences],
            "citations": {"enabled": True},
        }
        for src in sources
    ]
    content.append({"type": "text", "text": question})
    return {
        "model": settings.answer_model,
        "max_tokens": settings.answer_max_tokens,
        "system": [{"type": "text", "text": ANSWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": settings.answer_effort},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "messages": [{"role": "user", "content": content}],
    }


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class StreamTranslator:
    """Raw Messages stream events → AnswerEvents. Accepts SDK objects or plain dicts
    (the latter keeps unit tests free of SDK internals)."""

    def __init__(self, mapper: CitationMapper) -> None:
        self.mapper = mapper
        self.block_types: dict[int, str] = {}
        self.text_block_count = 0
        self.block_numbers: dict[int, int] = {}  # stream index → answer block number

    def feed(self, event: Any) -> list[AnswerEvent]:
        etype = _attr(event, "type")
        if etype == "content_block_start":
            index = _attr(event, "index")
            block = _attr(event, "content_block")
            btype = _attr(block, "type")
            self.block_types[index] = btype
            if btype == "thinking":
                return [AnswerEvent("status", {"phase": "thinking"})]
            if btype == "text":
                number = self.text_block_count
                self.text_block_count += 1
                self.block_numbers[index] = number
                out = [AnswerEvent("status", {"phase": "writing"})] if number == 0 else []
                out.append(AnswerEvent("block_start", {"block": number}))
                initial = _attr(block, "text") or ""
                if initial:
                    out.append(AnswerEvent("text", {"block": number, "text": initial}))
                return out
            if btype == "fallback":
                src, dst = _attr(block, "from_") or _attr(block, "from"), _attr(block, "to")
                return [
                    AnswerEvent(
                        "served_by",
                        {"from_model": _attr(src, "model"), "to_model": _attr(dst, "model")},
                    )
                ]
            return []
        if etype == "content_block_delta":
            index = _attr(event, "index")
            if index not in self.block_numbers:
                return []  # thinking / signature deltas: nothing to show
            number = self.block_numbers[index]
            delta = _attr(event, "delta")
            dtype = _attr(delta, "type")
            if dtype == "text_delta":
                return [AnswerEvent("text", {"block": number, "text": _attr(delta, "text") or ""})]
            if dtype == "citations_delta":
                mapped = self.mapper.map(_attr(delta, "citation"))
                if mapped is None:
                    return []
                return [AnswerEvent("citation", {"block": number, **mapped.to_dict()})]
            return []
        if etype == "content_block_stop":
            index = _attr(event, "index")
            if index in self.block_numbers:
                return [AnswerEvent("block_stop", {"block": self.block_numbers[index]})]
        return []


def _final_events(message: Any) -> list[AnswerEvent]:
    stop_reason = _attr(message, "stop_reason")
    usage = _attr(message, "usage")
    if stop_reason == "refusal":
        details = _attr(message, "stop_details")
        return [
            AnswerEvent(
                "refusal",
                {
                    "category": _attr(details, "category"),
                    "explanation": _attr(details, "explanation"),
                },
            )
        ]
    data = {
        "stop_reason": stop_reason,
        "model": _attr(message, "model"),
        "usage": {
            "input_tokens": _attr(usage, "input_tokens"),
            "output_tokens": _attr(usage, "output_tokens"),
            "cache_read_input_tokens": _attr(usage, "cache_read_input_tokens"),
        },
    }
    return [AnswerEvent("done", data)]


def _error_event(exc: Exception) -> AnswerEvent:
    if is_missing_credentials(exc):
        return AnswerEvent("error", {"kind": "credentials", "message": NO_CREDENTIALS_MESSAGE})
    if isinstance(exc, anthropic.APIStatusError):
        return AnswerEvent(
            "error", {"kind": "api", "status": exc.status_code, "message": str(exc.message)[:300]}
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return AnswerEvent(
            "error", {"kind": "connection", "message": "could not reach the Claude API"}
        )
    return AnswerEvent("error", {"kind": "internal", "message": type(exc).__name__})


def stream_answer(
    question: str, sources: list[SourceSection], settings: Settings | None = None
) -> Iterator[AnswerEvent]:
    translator = StreamTranslator(CitationMapper(sources))
    try:
        with get_client().beta.messages.stream(
            **build_request(question, sources, settings)
        ) as stream:
            for event in stream:
                yield from translator.feed(event)
            final = stream.get_final_message()
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        yield _error_event(exc)
        return
    except TypeError as exc:
        if not is_missing_credentials(exc):
            raise
        yield _error_event(exc)
        return
    yield from _final_events(final)


async def astream_answer(
    question: str, sources: list[SourceSection], settings: Settings | None = None
) -> AsyncIterator[AnswerEvent]:
    """Async variant for the API. If the consumer stops iterating (client disconnect),
    leaving the `async with` closes the upstream HTTP stream, so generation stops too."""
    translator = StreamTranslator(CitationMapper(sources))
    try:
        async with get_async_client().beta.messages.stream(
            **build_request(question, sources, settings)
        ) as stream:
            async for event in stream:
                for out in translator.feed(event):
                    yield out
            final = await stream.get_final_message()
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        yield _error_event(exc)
        return
    except TypeError as exc:
        if not is_missing_credentials(exc):
            raise
        yield _error_event(exc)
        return
    for out in _final_events(final):
        yield out


@dataclass
class AnswerBlock:
    text: str = ""
    citations: list[MappedCitation] = field(default_factory=list)


@dataclass
class CollectedAnswer:
    """The answer reassembled from events — what gates and confidence scoring read."""

    blocks: list[AnswerBlock] = field(default_factory=list)
    refused: bool = False
    refusal_category: str | None = None
    error: dict[str, Any] | None = None
    served_by: dict[str, Any] | None = None
    done: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks)

    @property
    def citations(self) -> list[MappedCitation]:
        return [c for b in self.blocks for c in b.citations]

    def apply(self, event: AnswerEvent) -> None:
        d = event.data
        if event.type == "block_start":
            while len(self.blocks) <= d["block"]:
                self.blocks.append(AnswerBlock())
        elif event.type == "text":
            self.blocks[d["block"]].text += d["text"]
        elif event.type == "citation":
            fields = {k: v for k, v in d.items() if k != "block"}
            self.blocks[d["block"]].citations.append(MappedCitation(**fields))
        elif event.type == "refusal":
            self.refused = True
            self.refusal_category = d.get("category")
            self.blocks = []  # partial text from a refused turn must not be shown or scored
        elif event.type == "served_by":
            self.served_by = d
        elif event.type == "error":
            self.error = d
        elif event.type == "done":
            self.done = d
