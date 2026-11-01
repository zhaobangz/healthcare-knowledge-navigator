"""Runtime configuration.

Every knob the eval ablation ladder turns lives here, so a run's configuration is one
object that can be serialized next to its results. Values come from `HKN_*` environment
variables or `backend/.env`; the Anthropic key is read by the SDK itself
(`ANTHROPIC_API_KEY` or an `ant auth login` profile) and never passes through here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HKN_",
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://hkn:hkn@localhost:5434/hkn"
    data_dir: Path = REPO_ROOT / "data"

    # --- Claude (one model per stage; all default to the current Opus) -------------------
    answer_model: str = "claude-opus-5-5"
    plan_model: str = "claude-opus-5-5"
    context_model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    answer_effort: str = "medium"
    plan_effort: str = "low"
    context_effort: str = "low"
    judge_effort: str = "medium"
    answer_max_tokens: int = 16000
    # "claude" (default) or "extractive" (no model: quote top source sentences verbatim).
    answer_mode: str = "claude"

    # --- local models --------------------------------------------------------------------
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    # BGE v1.5 retrieval instruction, prepended to queries only (not passages).
    embed_query_instruction: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    device: str | None = None  # None lets sentence-transformers pick (mps on Apple silicon)

    # --- chunking --------------------------------------------------------------------------
    chunk_target_tokens: int = 250
    chunk_overlap_sentences: int = 1
    section_window_tokens: int = 1500

    # --- retrieval ------------------------------------------------------------------------
    dense_candidates: int = 50
    fts_candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    top_k: int = 8
    hnsw_ef_search: int = 100
    # Soft (model-extracted) filters are dropped when they leave fewer than this many hits.
    soft_filter_min_hits: int = 5

    # --- feature flags (the ablation ladder) ------------------------------------------------
    enable_query_plan: bool = True
    enable_hyde: bool = True
    enable_rerank: bool = True
    enable_sub_questions: bool = True
    use_context_prefix: bool = True

    # --- gates ---------------------------------------------------------------------------
    # Gate 1: minimum top rerank logit to attempt an answer. bge-reranker-base logits sit
    # well above 0 for on-topic passages and far below for unrelated ones; the default is
    # deliberately permissive and is re-tuned on the dev split by `hkn eval calibrate`.
    gate_min_rerank_logit: float = -3.0
    confidence_weights_path: Path = BACKEND_ROOT / "confidence_weights.json"

    # --- NCBI E-utilities (PMC ingest) -------------------------------------------------------
    ncbi_tool: str = "healthcare-knowledge-navigator"
    ncbi_email: str = ""
    ncbi_api_key: str = ""

    # --- API -------------------------------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "seed_manifest.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
