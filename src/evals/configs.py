"""Retrieval configurations and the grid builder."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

RETRIEVERS = ("vector", "keyword", "hybrid")
TRANSFORMS = ("none", "rewrite", "hyde")


@dataclass(frozen=True)
class RetrievalConfig:
    retriever: str = "hybrid"
    chunk_chars: int = 2400
    chunk_overlap: int = 300
    top_k: int = 5
    rerank: bool = False
    query_transform: str = "none"   # none | rewrite (keyword-rich rewrite) | hyde (hypothetical answer)

    def __post_init__(self):
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}")
        if self.query_transform not in TRANSFORMS:
            raise ValueError(f"query_transform must be one of {TRANSFORMS}")
        if not 1 <= self.top_k <= 50 or self.chunk_chars < 200 or not 0 <= self.chunk_overlap < self.chunk_chars:
            raise ValueError("invalid top_k / chunk settings")

    @property
    def key(self) -> str:
        parts = [self.retriever, f"chunk {self.chunk_chars}/{self.chunk_overlap}", f"k={self.top_k}"]
        if self.query_transform != "none":
            parts.append(self.query_transform)
        if self.rerank:
            parts.append("rerank")
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def grid(retrievers=("hybrid",), chunk_sizes=((2400, 300),), top_k=(5,), rerank=(False,),
         query_transforms=("none",)) -> list[RetrievalConfig]:
    """Cartesian product of the chosen options (duplicates removed, order kept)."""
    out, seen = [], set()
    for r, (c, o), k, rr, t in itertools.product(retrievers, chunk_sizes, top_k, rerank, query_transforms):
        cfg = RetrievalConfig(retriever=r, chunk_chars=c, chunk_overlap=o, top_k=k, rerank=rr, query_transform=t)
        if cfg.key not in seen:
            seen.add(cfg.key)
            out.append(cfg)
    return out
