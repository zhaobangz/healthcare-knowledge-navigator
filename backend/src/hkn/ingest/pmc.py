"""PubMed Central Open Access ingester (NCBI E-utilities).

Etiquette (https://www.ncbi.nlm.nih.gov/books/NBK25497/): every request carries `tool`
and `email`; at most 3 requests/second, or 10 with an API key; efetch is batched by POST.
The search already asks for CC BY / CC0 articles, but the license is re-checked on every
article's own XML by `parse_jats`, which rejects anything else before storage.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from hkn.config import Settings, get_settings
from hkn.ingest.fetch import sha256_bytes
from hkn.ingest.parse_html import ParseError
from hkn.ingest.parse_jats import LicenseRejected, journal_title, parse_jats, pmcid
from hkn.ingest.parsed import ParsedDocument

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
LICENSE_FILTER = '("cc by license"[filter] OR "cc0 license"[filter])'
# Topic queries alone return a lot of bench science; keep clinical evidence.
CLINICAL_FILTER = (
    '(randomized[tiab] OR "systematic review"[tiab] OR "meta-analysis"[tiab] '
    'OR guideline*[tiab] OR "cohort study"[tiab] OR "clinical trial"[tiab])'
)
_ARTICLE = re.compile(rb"<article[\s>].*?</article>", re.S)


@dataclass
class PmcArticle:
    pmcid: str
    xml: bytes
    sha256: str
    journal: str
    parsed: ParsedDocument


class EUtils:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.interval = 0.11 if self.settings.ncbi_api_key else 0.34
        self._last = 0.0
        self.client = httpx.Client(timeout=60, headers={"User-Agent": self.settings.ncbi_tool})

    def _params(self, **extra: str) -> dict[str, str]:
        params = {"tool": self.settings.ncbi_tool, **extra}
        if self.settings.ncbi_email:
            params["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _wait(self) -> None:
        delay = self.interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def esearch(self, term: str, *, retmax: int) -> list[str]:
        self._wait()
        r = self.client.get(
            f"{EUTILS}/esearch.fcgi",
            params=self._params(db="pmc", term=term, retmax=str(retmax), sort="relevance"),
        )
        r.raise_for_status()
        return re.findall(r"<Id>(\d+)</Id>", r.text)

    def efetch(self, ids: list[str]) -> bytes:
        self._wait()
        r = self.client.post(
            f"{EUTILS}/efetch.fcgi", data=self._params(db="pmc", id=",".join(ids), retmode="xml")
        )
        r.raise_for_status()
        return r.content


def build_query(topic_query: str, *, since: int = 2015) -> str:
    return f"({topic_query}) AND {CLINICAL_FILTER} AND {LICENSE_FILTER} AND {since}:3000[pdat]"


def fetch_articles(
    topic_query: str, *, limit: int, batch: int = 100, settings: Settings | None = None
) -> Iterator[PmcArticle | tuple[str, str]]:
    """Yield parsed articles, or (id, reason) for articles that were skipped."""
    settings = settings or get_settings()
    api = EUtils(settings)
    ids = api.esearch(build_query(topic_query), retmax=limit)
    cache = settings.data_dir / "pmc"
    cache.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        missing = [i for i in chunk if not (cache / f"PMC{i}.xml").exists()]
        if missing:
            payload = api.efetch(missing)
            for raw in _ARTICLE.findall(payload):
                soup = BeautifulSoup(raw, "lxml-xml")
                pid = pmcid(soup)
                if pid:
                    (cache / f"{pid}.xml").write_bytes(raw)
        for i in chunk:
            path = cache / f"PMC{i}.xml"
            if not path.exists():
                yield (f"PMC{i}", "no XML returned (not in the OA subset)")
                continue
            raw = path.read_bytes()
            try:
                parsed = parse_jats(raw.decode("utf-8", errors="replace"))
            except LicenseRejected as exc:
                yield (f"PMC{i}", str(exc))
                continue
            except ParseError as exc:
                yield (f"PMC{i}", f"unparseable: {exc}")
                continue
            soup = BeautifulSoup(raw, "lxml-xml")
            yield PmcArticle(f"PMC{i}", raw, sha256_bytes(raw), journal_title(soup), parsed)
