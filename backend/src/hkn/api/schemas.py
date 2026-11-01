from __future__ import annotations

from pydantic import BaseModel, Field

from hkn.retrieval.search import Filters


class FilterBody(BaseModel):
    doc_types: list[str] = Field(default_factory=list)
    orgs: list[str] = Field(default_factory=list)
    specialties: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    exclude_superseded: bool = False

    def to_filters(self) -> Filters:
        return Filters(
            doc_types=self.doc_types or None,
            orgs=self.orgs or None,
            specialties=self.specialties or None,
            year_min=self.year_min,
            year_max=self.year_max,
            exclude_superseded=self.exclude_superseded,
        )


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: FilterBody = Field(default_factory=FilterBody)


class DocumentOut(BaseModel):
    id: int
    source_key: str
    title: str
    org: str
    doc_type: str
    specialty: list[str]
    pub_year: int | None
    url: str
    license: str
    authors: str
    superseded: bool
    sections: int
    chunks: int


class FacetsOut(BaseModel):
    doc_types: list[str]
    orgs: list[str]
    specialties: list[str]
    year_min: int | None
    year_max: int | None
    documents: int
    chunks: int


class SentenceOut(BaseModel):
    id: int
    ord: int
    text: str
    kind: str
    char_start: int
    char_end: int


class SectionOut(BaseModel):
    id: int
    document_id: int
    heading_path: list[str]
    anchor: str | None
    text: str
    sentences: list[SentenceOut]
