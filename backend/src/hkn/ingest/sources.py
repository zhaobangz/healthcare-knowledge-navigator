"""Seed manifest loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hkn.models import DOC_TYPES


@dataclass
class SourceSpec:
    source_key: str
    title: str
    org: str
    doc_type: str
    url: str
    license: str
    parser: dict[str, Any]
    specialty: list[str] = field(default_factory=list)
    pub_year: int | None = None
    version: str = ""
    display_url: str | None = None
    superseded: bool = False

    @property
    def kind(self) -> str:
        return str(self.parser.get("kind", "html"))

    @property
    def link(self) -> str:
        return self.display_url or self.url


@dataclass
class Manifest:
    user_agent: str
    documents: list[SourceSpec]
    pmc_topics: list[dict[str, str]]


def load_manifest(path: Path) -> Manifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    docs: list[SourceSpec] = []
    seen: set[str] = set()
    for entry in raw.get("documents", []):
        spec = SourceSpec(**entry)
        if spec.doc_type not in DOC_TYPES:
            raise ValueError(f"{spec.source_key}: unknown doc_type {spec.doc_type!r}")
        if spec.source_key in seen:
            raise ValueError(f"duplicate source_key {spec.source_key!r}")
        if not spec.license:
            raise ValueError(f"{spec.source_key}: every source must record its license")
        seen.add(spec.source_key)
        docs.append(spec)
    return Manifest(
        user_agent=defaults.get("user_agent", "hkn-research-bot"),
        documents=docs,
        pmc_topics=list(raw.get("pmc_topics", [])),
    )
