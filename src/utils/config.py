"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/retrieval_lab"
    docqa_path: Path = PROJECT_ROOT.parent / "document-qa-citations"  # the project under evaluation
    model: str = "claude-opus-5-5"
    effort: str = "low"                    # query rewriting / question writing are light tasks
    voyage_model: str = "voyage-3.5"
    embedding_dim: int = 1024
    rerank_model: str = "rerank-2.5"
    rerank_candidates: int = 30            # retrieve this many, rerank, keep top_k
    # Prices in USD per million tokens, for the cost column. Claude Opus 5.5 is $4 in / $20 out;
    # set the Voyage prices from your plan (left at 0 rather than guessed).
    claude_input_price: float = 4.0
    claude_output_price: float = 20.0
    voyage_embed_price: float = 0.0
    voyage_rerank_price: float = 0.0
    max_parallel_configs: int = 4
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL (pgvector is required).")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    docqa = os.getenv("DOCQA_PATH")
    settings = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        docqa_path=(PROJECT_ROOT / docqa).resolve() if docqa and not Path(docqa).is_absolute()
        else Path(docqa) if docqa else Settings.docqa_path,
        model=os.getenv("AGENT_MODEL", Settings.model),
        voyage_model=os.getenv("VOYAGE_MODEL", Settings.voyage_model),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", Settings.embedding_dim)),
        rerank_model=os.getenv("VOYAGE_RERANK_MODEL", Settings.rerank_model),
        voyage_embed_price=float(os.getenv("VOYAGE_EMBED_PRICE_PER_MTOK", Settings.voyage_embed_price)),
        voyage_rerank_price=float(os.getenv("VOYAGE_RERANK_PRICE_PER_MTOK", Settings.voyage_rerank_price)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(settings, **{k: v for k, v in overrides.items() if v is not None})
