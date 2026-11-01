"""Query planning: rewriting, keyword expansion, decomposition, filters and HyDE in ONE
structured call (concepts 8, 9, 10).

These are separate techniques, but four serial model calls before retrieval would make
every answer slow; one low-effort call returns them all. Planning is an enhancement, not
a safety gate, so it fails *soft*: on an API error or refusal, retrieval proceeds with
the raw question and the trace records why.
"""

from __future__ import annotations

import logging
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from hkn.config import Settings, get_settings
from hkn.llm import get_client, is_missing_credentials

log = logging.getLogger(__name__)

DocTypeLiteral = Literal[
    "guideline", "protocol", "systematic_review", "rct", "review", "drug_label"
]
SpecialtyLiteral = Literal["infectious_disease", "hepatology", "pain_management", "oncology"]


class SoftFilters(BaseModel):
    """Filters the model *infers* from the question. Applied first, dropped if they starve
    retrieval — unlike filters the user sets explicitly in the UI, which always hold."""

    doc_types: list[DocTypeLiteral] = Field(default_factory=list)
    specialties: list[SpecialtyLiteral] = Field(default_factory=list)
    year_min: int | None = None


class QueryPlan(BaseModel):
    is_clinical: bool = Field(description="False for questions unrelated to clinical medicine.")
    asks_personal_advice: bool = Field(
        description="True if the user asks what THEY personally should do about their own health."
    )
    rewritten_query: str = Field(
        description="The question restated as a precise, self-contained clinical search query."
    )
    keywords: list[str] = Field(
        description="3-12 short keyword phrases for full-text search: key terms, synonyms, "
        "generic drug names, abbreviations spelled out."
    )
    sub_questions: list[str] = Field(
        description="0-3 simpler questions, only if the question has several distinct parts."
    )
    soft_filters: SoftFilters
    hyde_passage: str = Field(
        description="2-4 sentences written as if quoted from a clinical guideline that "
        "answers the question. Used only as a search probe, never shown to users."
    )


PLANNER_SYSTEM = """You plan searches over a clinical knowledge base. The corpus holds US \
clinical guidelines (CDC MMWR recommendations, USPSTF recommendation statements), NCI \
treatment summaries for health professionals, FDA drug labels (DailyMed), and open-access \
research papers (PubMed Central). Specialties covered: infectious disease (STIs, doxycycline \
PEP), hepatology (hepatitis B and C screening), pain management (opioid prescribing), and \
oncology (breast and colon cancer treatment and screening).

Produce a search plan for the user's question. Keywords should be what the relevant passage \
would literally contain. Infer soft filters only when the question clearly implies them (for \
example "according to the label" implies drug_label; "screening recommendation" implies \
guideline). Leave filters empty when unsure. The HyDE passage is a search probe: write it in \
the register of a guideline, with plausible specifics, even if you are unsure of the facts."""


def plan_query(
    question: str, *, settings: Settings | None = None
) -> tuple[QueryPlan | None, str | None]:
    """Return (plan, error). `plan` is None when planning failed or was refused."""
    settings = settings or get_settings()
    try:
        response = get_client().messages.parse(
            model=settings.plan_model,
            max_tokens=4000,
            system=PLANNER_SYSTEM,
            output_config={"effort": settings.plan_effort},  # type: ignore[typeddict-item]
            messages=[{"role": "user", "content": question}],
            output_format=QueryPlan,
        )
    except anthropic.APIStatusError as exc:
        log.warning("query planning failed: %s", exc)
        return None, f"planner API error {exc.status_code}"
    except anthropic.APIConnectionError as exc:
        log.warning("query planning failed: %s", exc)
        return None, "planner connection error"
    except TypeError as exc:
        if is_missing_credentials(exc):
            return None, "no Anthropic credentials configured"
        raise
    if response.stop_reason == "refusal":
        return None, "planner declined"
    if response.parsed_output is None:
        return None, f"planner returned no plan (stop_reason={response.stop_reason})"
    return response.parsed_output, None
