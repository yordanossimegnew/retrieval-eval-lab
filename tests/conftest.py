"""Fixtures: a dedicated PostgreSQL + pgvector test database and the Document QA sample files."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

from src.agent import Lab, open_checkpointer
from src.models import Usage
from src.tools import IndexManager, LabDB
from src.utils.config import PROJECT_ROOT, Settings
from tests.fakes import DIM, FakeLabModels, HashEmbedder, OverlapReranker

load_dotenv(PROJECT_ROOT / ".env")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
DOCQA = (PROJECT_ROOT / os.getenv("DOCQA_PATH", "../document-qa-citations")).resolve()
SAMPLES = DOCQA / "data" / "samples"


def pytest_collection_modifyitems(config, items):
    reasons = []
    if not TEST_DATABASE_URL:
        reasons.append("TEST_DATABASE_URL is not set")
    if not (DOCQA / "src" / "__init__.py").exists():
        reasons.append(f"document-qa-citations not found at {DOCQA}")
    if reasons:
        skip = pytest.mark.skip(reason="; ".join(reasons))
        for item in items:
            if "settings" in item.fixturenames:
                item.add_marker(skip)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_url=TEST_DATABASE_URL, docqa_path=DOCQA, embedding_dim=DIM, log_dir=tmp_path / "logs",
                    rerank_candidates=10, voyage_embed_price=0.06, voyage_rerank_price=0.05)


@pytest.fixture
def db(settings):
    lab_db = LabDB(settings.database_url)
    lab_db.init()
    with lab_db.engine.begin() as c:
        for (schema,) in c.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'idx_%'")):
            c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        c.execute(text("TRUNCATE corpora, corpus_docs, question_sets, questions, indexes, runs, run_results, run_queries"))
    yield lab_db
    lab_db.engine.dispose()


@pytest.fixture
def lab(settings, db):
    usage = Usage()
    saver = open_checkpointer(settings.database_url)
    embedder = HashEmbedder()
    yield Lab(settings, db, IndexManager(db, settings.docqa_path, embedder, "hash-test", DIM), embedder,
              OverlapReranker(), FakeLabModels(), usage, saver)
    saver.conn.close()


@pytest.fixture
def sample_files() -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(SAMPLES.iterdir()) if p.is_file()]
