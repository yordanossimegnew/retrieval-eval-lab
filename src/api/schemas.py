"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    dataset: Literal["scifact"] = "scifact"
    max_queries: int = Field(100, ge=5, le=300)
    max_docs: int = Field(800, ge=50, le=5183)


class SyntheticRequest(BaseModel):
    corpus_id: str
    name: str = Field("Synthetic questions", max_length=200)
    passages: int = Field(20, ge=1, le=300, description="Passages to sample across the corpus.")
    per_passage: int = Field(2, ge=1, le=5, description="Questions Claude may write per passage.")


class GridRequest(BaseModel):
    retrievers: list[Literal["vector", "keyword", "hybrid"]] = ["vector", "keyword", "hybrid"]
    chunk_sizes: list[tuple[int, int]] = Field([(2400, 300)], description="(chunk_chars, chunk_overlap) pairs")
    top_k: list[int] = [5]
    rerank: list[bool] = [False]
    query_transforms: list[Literal["none", "rewrite", "hyde"]] = ["none"]


class RunRequest(BaseModel):
    question_set_id: str
    name: str = Field("Experiment", max_length=200)
    grid: GridRequest = GridRequest()


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embeddings: str
    reranker: str
    docqa_path: str
    database: str
