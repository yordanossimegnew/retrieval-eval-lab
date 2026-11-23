"""Models used by the lab, all via LangChain, with token accounting for the cost column.

* Embeddings:  VoyageAIEmbeddings (same model as Document QA by default)
* Reranking:   VoyageAIRerank (a LangChain document compressor)
* LLM:         ChatAnthropic - question writing (structured output), query rewrite, HyDE
"""

from __future__ import annotations

import threading
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from pydantic import BaseModel, Field

from ..prompts import HYDE_PROMPT, QUESTION_PROMPT, REWRITE_PROMPT
from ..utils.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Usage:
    """Thread-safe token counters (parallel configs share one model instance)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts = {"llm_input": 0, "llm_output": 0, "embed": 0, "rerank": 0}

    def add(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                self.counts[k] += int(v or 0)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.counts)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]: ...


class LabModels(Protocol):
    def write_questions(self, title: str, passage: str, n: int) -> list[dict]: ...
    def transform_query(self, question: str, mode: str) -> str: ...


def make_embedder(settings: Settings) -> Embedder:
    return VoyageAIEmbeddings(model=settings.voyage_model, batch_size=64)


class VoyageReranker:
    def __init__(self, settings: Settings, usage: Usage | None = None):
        self.model = settings.rerank_model
        self.label = f"voyage/{settings.rerank_model}"
        self.usage = usage or Usage()

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        compressor = VoyageAIRerank(model=self.model, top_k=top_k)
        docs = [Document(page_content=c["text"], metadata={"i": i}) for i, c in enumerate(chunks)]
        ranked = compressor.compress_documents(docs, query)
        self.usage.add(rerank=sum(len(c["text"]) // 4 for c in chunks) + len(query) // 4)
        return [{**chunks[d.metadata["i"]], "rerank_score": float(d.metadata.get("relevance_score", 0))} for d in ranked]


class GeneratedQuestion(BaseModel):
    question: str = Field(description="A natural question this passage answers.")
    evidence: str = Field(description="An exact quote from the passage (one sentence or less) that answers it.")


class GeneratedQuestions(BaseModel):
    questions: list[GeneratedQuestion]


class AnthropicLabModels:
    def __init__(self, settings: Settings, usage: Usage | None = None):
        self.usage = usage or Usage()
        self.label = settings.model
        self.chat = ChatAnthropic(model=settings.model, max_tokens=4000, output_config={"effort": settings.effort},
                                  betas=[FALLBACK_BETA], model_kwargs={"fallbacks": "default"}, max_retries=3)

    def _track(self, message) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        self.usage.add(llm_input=meta.get("input_tokens"), llm_output=meta.get("output_tokens"))

    def write_questions(self, title: str, passage: str, n: int) -> list[dict]:
        model = self.chat.with_structured_output(GeneratedQuestions, method="json_schema", include_raw=True)
        out = model.invoke([SystemMessage(QUESTION_PROMPT.format(n=n)),
                            HumanMessage(f"<document_title>{title}</document_title>\n<passage>\n{passage}\n</passage>")])
        self._track(out["raw"])
        parsed = out["parsed"]
        return [q.model_dump() for q in parsed.questions] if parsed else []

    def transform_query(self, question: str, mode: str) -> str:
        if mode == "none":
            return question
        prompt = REWRITE_PROMPT if mode == "rewrite" else HYDE_PROMPT
        message = self.chat.invoke([SystemMessage(prompt), HumanMessage(question)])
        self._track(message)
        text = message.content if isinstance(message.content, str) else "".join(
            b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text")
        return text.strip() or question
