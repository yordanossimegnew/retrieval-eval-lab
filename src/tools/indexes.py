"""Build one Document QA index per (corpus, chunk size, overlap, embedding model).

Each index lives in its own Postgres schema with Document QA's exact tables (documents,
chunks with pgvector + tsvector), built with Document QA's own loader, chunker and store.
Indexes are cached: re-running an experiment with the same chunking reuses them.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from sqlalchemy import text

from .docqa import load_docqa
from .labdb import LabDB, engine_url

_build_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def schema_name(corpus_id: str, chunk_chars: int, overlap: int, embed_model: str) -> str:
    digest = hashlib.sha1(f"{corpus_id}|{chunk_chars}|{overlap}|{embed_model}".encode()).hexdigest()[:10]
    return f"idx_{digest}"


def estimate_tokens(texts: list[str]) -> int:
    return sum(max(1, len(t) // 4) for t in texts)


class IndexManager:
    def __init__(self, db: LabDB, docqa_path: str, embedder: Any, embed_model: str, embedding_dim: int):
        self.db, self.docqa = db, load_docqa(str(docqa_path))
        self.embedder, self.embed_model, self.dim = embedder, embed_model, embedding_dim
        self._stores: dict[str, Any] = {}

    def store(self, schema: str):
        """A Document QA DocumentStore bound to one index schema."""
        if schema not in self._stores:
            url = engine_url(self.db.url, search_path=f"{schema},public").replace("postgresql+psycopg://", "postgresql://")
            self._stores[schema] = self.docqa.DocumentStore(url, self.dim)
        return self._stores[schema]

    def ensure(self, corpus_id: str, chunk_chars: int, overlap: int, on_progress=None) -> dict:
        schema = schema_name(corpus_id, chunk_chars, overlap, self.embed_model)
        with _guard:
            lock = _build_locks.setdefault(schema, threading.Lock())
        with lock:  # parallel configs sharing a chunking build it once
            existing = self.db.index_row(schema)
            if existing:
                return existing
            return self._build(schema, corpus_id, chunk_chars, overlap, on_progress)

    def _build(self, schema: str, corpus_id: str, chunk_chars: int, overlap: int, on_progress) -> dict:
        start = time.monotonic()
        with self.db.engine.begin() as c:
            c.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            c.execute(text(f'CREATE SCHEMA "{schema}"'))
        store = self.store(schema)
        store.init()
        dq = self.docqa
        docs = self.db.corpus_documents(corpus_id)
        total_chunks, tokens = 0, 0
        for i, row in enumerate(docs, 1):
            if row["raw"] is not None:
                loaded = dq.load_file(row["filename"], bytes(row["raw"]))
            else:
                loaded = dq.LoadedDocument(title=row["title"], source_type="text", source=row["doc_key"],
                                           sections=[dq.Section(text=row["text"])])
            loaded.source = row["doc_key"]  # the key questions use to name relevant documents
            chunks = dq.chunk_sections(loaded.sections, chunk_chars, overlap)
            if not chunks:
                continue
            texts = [f"{loaded.title}\n{c.heading or ''}\n{c.text}".strip() for c in chunks]  # as Document QA does
            vectors = self.embedder.embed_documents(texts)
            store.add(loaded, chunks, vectors)
            total_chunks += len(chunks)
            tokens += estimate_tokens(texts)
            if on_progress and (i % 25 == 0 or i == len(docs)):
                on_progress({"stage": f"indexing chunk={chunk_chars}/{overlap}", "docs": i, "of": len(docs)})
        values = {"schema": schema, "corpus_id": corpus_id, "chunk_chars": chunk_chars, "chunk_overlap": overlap,
                  "embed_model": self.embed_model, "chunks": total_chunks, "embed_tokens": tokens,
                  "build_seconds": round(time.monotonic() - start, 2)}
        self.db.save_index(**values)
        return self.db.index_row(schema)
