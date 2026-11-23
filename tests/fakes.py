"""Offline stand-ins for Voyage (embeddings, rerank) and Claude (question writing, query transforms)."""

from __future__ import annotations

import hashlib
import math
import re

DIM = 1024


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]


class HashEmbedder:
    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Scores by query-word overlap density, like a crude cross-encoder."""

    label = "local-overlap (stand-in for Voyage rerank)"

    def rerank(self, query, chunks, top_k):
        q = set(words(query))
        scored = [{**c, "rerank_score": len(q & set(words(c["text"]))) / (1 + len(words(c["text"])) ** 0.5)} for c in chunks]
        return sorted(scored, key=lambda c: -c["rerank_score"])[:top_k]


class FakeLabModels:
    """Writes one question per passage from its first factual sentence; also emits one invalid
    question (evidence not in the passage) so the validation filter has something to drop."""

    label = "scripted (stand-in for Claude)"

    def __init__(self):
        self.transform_calls = 0

    def write_questions(self, title, passage, n):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.strip()) > 30]
        if not sentences:
            return []
        evidence = sentences[0]
        subject = " ".join(words(evidence)[:6])
        return [
            {"question": f"What does {title} say about {subject}?", "evidence": evidence},
            {"question": "Is this invented evidence ever checked?", "evidence": "This sentence is not in the passage."},
        ][:max(1, n)]

    def transform_query(self, question, mode):
        if mode == "none":
            return question
        self.transform_calls += 1
        return question + " details explanation" if mode == "rewrite" else f"The document explains: {question}"
