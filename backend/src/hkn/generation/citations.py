"""Map Claude's `search_result_location` citations back to stored sentences.

A citation names a search result by its position in the request
(`search_result_index`) and a half-open range of that result's content blocks
(`start_block_index`..`end_block_index`). Because each content block is exactly one
stored sentence (see `retrieval/expand.py`), the range maps straight to sentence IDs and
their character offsets in the section text — the UI highlight needs nothing else.

A citation that points outside the request is a contract violation, not a highlight to
guess at: it is dropped and counted, so citation coverage drops and the confidence score
reflects it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote

from hkn.retrieval.expand import SourceSection


@dataclass
class MappedCitation:
    source_index: int  # search_result_index; the UI shows it as [source_index + 1]
    section_id: int
    document_id: int
    sentence_ids: list[int]
    cited_text: str
    title: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CitationMapper:
    sources: list[SourceSection]
    invalid: int = 0
    _seen: set[tuple[int, int, int]] = field(default_factory=set)

    def map(self, citation: Any) -> MappedCitation | None:
        """Accepts an SDK citation object or a plain dict with the same fields."""
        get = (
            (lambda k: citation.get(k))
            if isinstance(citation, dict)
            else (lambda k: getattr(citation, k, None))
        )
        if get("type") != "search_result_location":
            self.invalid += 1
            return None
        idx, start, end = (
            get("search_result_index"),
            get("start_block_index"),
            get("end_block_index"),
        )
        if not isinstance(idx, int) or not isinstance(start, int) or not isinstance(end, int):
            self.invalid += 1
            return None
        if not (0 <= idx < len(self.sources)):
            self.invalid += 1
            return None
        source = self.sources[idx]
        if not (0 <= start < end <= len(source.sentences)):
            self.invalid += 1
            return None
        sentences = source.sentences[start:end]
        return MappedCitation(
            source_index=idx,
            section_id=source.section_id,
            document_id=source.document_id,
            sentence_ids=[s.id for s in sentences],
            cited_text=str(get("cited_text") or " ".join(s.text for s in sentences)),
            title=source.title,
            url=text_fragment_url(source.source_url, sentences[0].text),
        )


def text_fragment_url(url: str, sentence: str, words: int = 8) -> str:
    """Deep link that scrolls to and highlights the cited sentence in browsers that
    support URL text fragments. The directive follows any existing anchor
    (`page#section:~:text=…`) or starts a fragment (`page#:~:text=…`)."""
    snippet = " ".join(sentence.split()[:words])
    if not snippet:
        return url
    directive = f":~:text={quote(snippet, safe='')}"
    return f"{url}{directive}" if "#" in url else f"{url}#{directive}"
