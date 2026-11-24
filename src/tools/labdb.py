"""The lab's own tables: corpora, question sets, runs and per-query results (PostgreSQL)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData()

corpora = Table(
    "corpora", metadata,
    Column("id", String(32), primary_key=True),
    Column("name", Text, nullable=False),
    Column("kind", String(20), nullable=False),       # files | beir
    Column("meta", JSONB, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
corpus_docs = Table(
    "corpus_docs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("corpus_id", String(32), ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("doc_key", Text, nullable=False),             # file name or "beir:<doc id>"
    Column("title", Text, nullable=False),
    Column("filename", Text),                            # files: original name (drives the loader)
    Column("raw", LargeBinary),                          # files: original bytes, re-chunked per config
    Column("text", Text),                                # benchmark docs: plain text
)
question_sets = Table(
    "question_sets", metadata,
    Column("id", String(32), primary_key=True),
    Column("corpus_id", String(32), ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("name", Text, nullable=False),
    Column("source", String(20), nullable=False),        # synthetic | benchmark
    Column("meta", JSONB, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
questions = Table(
    "questions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("set_id", String(32), ForeignKey("question_sets.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("question", Text, nullable=False),
    # [{"doc_key": "...", "evidence": "exact quote" | null}]  (evidence null = whole document is relevant)
    Column("relevant", JSONB, nullable=False),
    Column("meta", JSONB, nullable=False, default=dict),
)
indexes = Table(
    "indexes", metadata,
    Column("schema", String(40), primary_key=True),
    Column("corpus_id", String(32), ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_chars", Integer, nullable=False),
    Column("chunk_overlap", Integer, nullable=False),
    Column("embed_model", Text, nullable=False),
    Column("chunks", Integer, nullable=False),
    Column("embed_tokens", Integer, nullable=False),
    Column("build_seconds", Float, nullable=False),
    Column("built_at", DateTime(timezone=True), nullable=False),
)
runs = Table(
    "runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("set_id", String(32), ForeignKey("question_sets.id", ondelete="CASCADE"), nullable=False),
    Column("name", Text, nullable=False),
    Column("status", String(20), nullable=False),        # queued | running | completed | failed
    Column("configs", JSONB, nullable=False),
    Column("progress", JSONB, nullable=False, default=dict),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
)
run_results = Table(
    "run_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("config_key", Text, nullable=False),
    Column("config", JSONB, nullable=False),
    Column("metrics", JSONB, nullable=False),
    Column("latency", JSONB, nullable=False),
    Column("cost", JSONB, nullable=False),
)
run_queries = Table(
    "run_queries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("config_key", Text, nullable=False),
    Column("question_id", Integer, nullable=False),
    Column("query_used", Text, nullable=False),
    Column("metrics", JSONB, nullable=False),
    Column("retrieved", JSONB, nullable=False),
    Column("latency_ms", Float, nullable=False),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def engine_url(url: str, search_path: str | None = None) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg://" + url[len(prefix):]
    if search_path:
        url += ("&" if "?" in url else "?") + f"options=-csearch_path%3D{search_path}"
    return url


def _row(row) -> dict:
    out = dict(row._mapping)
    for k, v in out.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    out.pop("raw", None)
    return out


class LabDB:
    def __init__(self, database_url: str):
        self.url = database_url
        self.engine: Engine = create_engine(engine_url(database_url), pool_pre_ping=True, pool_size=8, max_overflow=8)

    def init(self) -> None:
        with self.engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        metadata.create_all(self.engine)

    # ------------------------------------------------------------------ corpora
    def add_corpus(self, name: str, kind: str, docs: list[dict], meta: dict | None = None) -> dict:
        cid = new_id()
        with self.engine.begin() as c:
            c.execute(insert(corpora).values(id=cid, name=name, kind=kind, meta=meta or {}, created_at=now()))
            if docs:
                c.execute(insert(corpus_docs), [{**d, "corpus_id": cid} for d in docs])
        return self.corpus(cid)

    def corpus(self, cid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(corpora).where(corpora.c.id == cid)).first()
            if row is None:
                return None
            n = c.execute(text("SELECT count(*) FROM corpus_docs WHERE corpus_id = :c"), {"c": cid}).scalar()
        return {**_row(row), "documents": n}

    def list_corpora(self) -> list[dict]:
        with self.engine.connect() as c:
            ids = [r.id for r in c.execute(select(corpora.c.id).order_by(corpora.c.created_at.desc()))]
        return [self.corpus(i) for i in ids]

    def corpus_documents(self, cid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(corpus_docs).where(corpus_docs.c.corpus_id == cid).order_by(corpus_docs.c.id)).all()
        return [dict(r._mapping) for r in rows]

    # ------------------------------------------------------------------ question sets
    def add_question_set(self, corpus_id: str, name: str, source: str, items: list[dict], meta: dict | None = None) -> dict:
        sid = new_id()
        with self.engine.begin() as c:
            c.execute(insert(question_sets).values(id=sid, corpus_id=corpus_id, name=name, source=source,
                                                   meta=meta or {}, created_at=now()))
            if items:
                c.execute(insert(questions), [{"set_id": sid, "question": q["question"], "relevant": q["relevant"],
                                               "meta": q.get("meta", {})} for q in items])
        return self.question_set(sid)

    def set_question_set_meta(self, sid: str, meta: dict) -> None:
        with self.engine.begin() as c:
            c.execute(update(question_sets).where(question_sets.c.id == sid).values(meta=meta))

    def question_set(self, sid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(question_sets).where(question_sets.c.id == sid)).first()
            if row is None:
                return None
            n = c.execute(text("SELECT count(*) FROM questions WHERE set_id = :s"), {"s": sid}).scalar()
        return {**_row(row), "questions": n}

    def list_question_sets(self) -> list[dict]:
        with self.engine.connect() as c:
            ids = [r.id for r in c.execute(select(question_sets.c.id).order_by(question_sets.c.created_at.desc()))]
        return [self.question_set(i) for i in ids]

    def questions_of(self, sid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(questions).where(questions.c.set_id == sid).order_by(questions.c.id)).all()
        return [dict(r._mapping) for r in rows]

    # ------------------------------------------------------------------ indexes
    def index_row(self, schema: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(indexes).where(indexes.c.schema == schema)).first()
        return _row(row) if row else None

    def save_index(self, **values) -> None:
        with self.engine.begin() as c:
            c.execute(delete(indexes).where(indexes.c.schema == values["schema"]))
            c.execute(insert(indexes).values(**values, built_at=now()))

    # ------------------------------------------------------------------ runs
    def create_run(self, set_id: str, name: str, configs: list[dict]) -> dict:
        rid = new_id()
        with self.engine.begin() as c:
            c.execute(insert(runs).values(id=rid, set_id=set_id, name=name, status="queued", configs=configs,
                                          progress={"done": 0, "total": len(configs), "stage": "queued"},
                                          created_at=now()))
        return self.run(rid)

    def update_run(self, rid: str, **values) -> None:
        if values.get("status") in ("completed", "failed"):
            values["finished_at"] = now()
        with self.engine.begin() as c:
            c.execute(update(runs).where(runs.c.id == rid).values(**values))

    def run(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(runs).where(runs.c.id == rid)).first()
            if row is None:
                return None
            results = c.execute(select(run_results).where(run_results.c.run_id == rid).order_by(run_results.c.id)).all()
        return {**_row(row), "results": [_row(r) for r in results]}

    def list_runs(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(runs).order_by(runs.c.created_at.desc())).all()
        return [_row(r) for r in rows]

    def save_result(self, rid: str, config_key: str, config: dict, metrics: dict, latency: dict, cost: dict,
                    per_query: list[dict]) -> None:
        with self.engine.begin() as c:
            c.execute(insert(run_results).values(run_id=rid, config_key=config_key, config=config, metrics=metrics,
                                                 latency=latency, cost=cost))
            if per_query:
                c.execute(insert(run_queries), [{**q, "run_id": rid, "config_key": config_key} for q in per_query])

    def run_queries_of(self, rid: str, config_key: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(run_queries, questions.c.question, questions.c.relevant)
                             .join(questions, questions.c.id == run_queries.c.question_id)
                             .where(run_queries.c.run_id == rid, run_queries.c.config_key == config_key)
                             .order_by(run_queries.c.question_id)).all()
        return [dict(r._mapping) for r in rows]
