"""End to end on PostgreSQL: corpus -> synthetic + benchmark questions -> indexes -> parallel runs."""

from __future__ import annotations

import io

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from src.agent import build_eval_graph, build_generation_graph
from src.api import create_app
from src.evals import grid
from src.evals.datasets import SCIFACT, file_documents, scifact_subset, validate_generated


def test_validation_keeps_only_exact_quotes():
    passage = {"doc_key": "a.md", "title": "A", "passage": "Passwords must be at least 14 characters long. Keys rotate."}
    kept, dropped = validate_generated(passage, [
        {"question": "How long do passwords need to be?", "evidence": "Passwords must be at least 14 characters long."},
        {"question": "What is the refund policy?", "evidence": "Refunds within 30 days."},
        {"question": "How long do passwords need to be?", "evidence": "Passwords must be at least 14 characters long."},
        {"question": "Keys?", "evidence": "Keys rotate."},
    ], set())
    assert [k["question"] for k in kept] == ["How long do passwords need to be?"]
    assert kept[0]["relevant"] == [{"doc_key": "a.md", "evidence": "Passwords must be at least 14 characters long."}]
    assert [d["reason"] for d in dropped] == ["evidence is not an exact quote from the passage", "duplicate question",
                                              "not a well-formed question"]


def _parquet(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def test_scifact_subset_from_mock_hub(tmp_path):
    corpus = [{"_id": str(i), "title": f"T{i}", "text": f"Abstract {i}"} for i in range(1, 21)]
    queries = [{"_id": "q1", "text": "Does X cause Y?"}, {"_id": "q2", "text": "Is Z safe?"}]
    qrels = "query-id\tcorpus-id\tscore\nq1\t3\t1\nq2\t7\t1\nq2\t8\t1\n"

    def handler(request):
        url = str(request.url)
        body = _parquet(corpus) if url == SCIFACT["corpus"] else _parquet(queries) if url == SCIFACT["queries"] else qrels.encode()
        return httpx.Response(200, content=body)

    docs, items = scifact_subset(tmp_path, max_queries=5, max_docs=10, client=httpx.Client(transport=httpx.MockTransport(handler)))
    keys = {d["doc_key"] for d in docs}
    assert len(docs) == 10 and {"beir:3", "beir:7", "beir:8"} <= keys  # relevant docs always included
    q2 = next(i for i in items if i["question"] == "Is Z safe?")
    assert q2["relevant"] == [{"doc_key": "beir:7", "evidence": None}, {"doc_key": "beir:8", "evidence": None}]
    assert (tmp_path / "test.tsv").exists()  # cached for next time


def test_generation_and_parallel_runs(lab, db, sample_files):
    corpus = db.add_corpus("Acme docs", "files", file_documents(sample_files, str(lab.settings.docqa_path)))
    qset = db.add_question_set(corpus["id"], "Synthetic", "synthetic", [], {"status": "generating"})
    build_generation_graph(lab).invoke(
        {"set_id": qset["id"], "corpus_id": corpus["id"], "n_passages": 6, "per_passage": 2},
        {"configurable": {"thread_id": f"gen-{qset['id']}"}})
    qset = db.question_set(qset["id"])
    assert qset["meta"]["status"] == "ready" and qset["questions"] >= 4
    assert qset["meta"]["dropped"] >= 4  # every invented-evidence question was rejected
    assert all(q["relevant"][0]["evidence"] for q in db.questions_of(qset["id"]))

    configs = grid(("vector", "keyword", "hybrid"), ((300, 50), (2400, 300)), (3,), (False, True), ("none", "rewrite"))
    run = db.create_run(qset["id"], "Grid", [c.to_dict() for c in configs])
    build_eval_graph(lab).invoke({"run_id": run["id"], "set_id": qset["id"], "corpus_id": corpus["id"],
                                  "configs": [c.to_dict() for c in configs]},
                                 {"configurable": {"thread_id": f"run-{run['id']}"}, "max_concurrency": 4})
    result = db.run(run["id"])
    assert result["status"] == "completed" and len(result["results"]) == len(configs) == 24
    by_key = {r["config_key"]: r for r in result["results"]}
    hybrid = by_key["hybrid · chunk 2400/300 · k=3"]["metrics"]
    assert hybrid["queries"] == qset["questions"] and hybrid["recall"] > 0.5
    assert 0 <= hybrid["precision"] <= 1 and hybrid["ndcg"] <= 1
    assert all(r["latency"]["p95_ms"] >= r["latency"]["p50_ms"] >= 0 for r in result["results"])
    reranked = by_key["hybrid · chunk 2400/300 · k=3 · rerank"]["cost"]
    assert reranked["rerank_tokens_est"] > 0 and reranked["usd"] > 0
    assert result["progress"]["index_cost"]["embed_tokens"] > 0

    # two chunkings -> exactly two index schemas, built once and shared by 12 configs each
    with db.engine.connect() as c:
        from sqlalchemy import text
        assert c.execute(text("SELECT count(*) FROM indexes")).scalar() == 2
    assert lab.models.transform_calls == qset["questions"]  # each question rewritten once, shared across configs

    drill = db.run_queries_of(run["id"], "keyword · chunk 300/50 · k=3")
    assert len(drill) == qset["questions"] and "retrieved" in drill[0] and "question" in drill[0]


def test_api_flow(settings, db, lab, sample_files):
    def scifact(cache, max_queries, max_docs):
        docs = [{"doc_key": f"beir:{i}", "title": f"Paper {i}", "filename": None, "raw": None,
                 "text": f"Paper {i}. " + ("Vitamin D lowers fracture risk in adults. " if i == 2 else "Unrelated cell biology. ")}
                for i in range(1, 6)]
        return docs, [{"question": "Does vitamin D reduce fractures?", "relevant": [{"doc_key": "beir:2", "evidence": None}]}]

    client = TestClient(create_app(settings, lab_factory=lambda _db: lab, scifact_loader=scifact, run_inline=True))
    assert client.get("/health").json()["status"] == "ok"

    up = client.post("/corpora", data={"name": "Acme"}, files=[("files", f) for f in sample_files])
    assert up.status_code == 201 and up.json()["documents"] == len(sample_files)
    bad = client.post("/corpora", data={"name": "x"}, files=[("files", ("evil.exe", b"MZ"))])
    assert bad.status_code == 422

    bench = client.post("/corpora/benchmark", json={"max_queries": 5, "max_docs": 50}).json()
    run = client.post("/runs", json={"question_set_id": bench["question_set"]["id"], "name": "SciFact",
                                     "grid": {"retrievers": ["vector", "keyword", "hybrid"], "chunk_sizes": [[800, 100]]}})
    assert run.status_code == 202
    detail = client.get(f"/runs/{run.json()['id']}").json()
    assert detail["status"] == "completed" and detail["progress"]["done"] == 3
    assert detail["results"][0]["metrics"]["recall"] == 1.0  # best config listed first

    keys = [r["config_key"] for r in detail["results"]]
    cmp = client.get(f"/runs/{run.json()['id']}/compare", params={"a": keys[0], "b": keys[-1]}).json()
    assert set(cmp) == {"a", "b", "only_a_found", "only_b_found"}

    syn = client.post("/question-sets/synthetic", json={"corpus_id": up.json()["id"], "passages": 3}).json()
    assert client.get(f"/question-sets/{syn['id']}").json()["meta"]["status"] == "ready"
    empty = db.add_question_set(up.json()["id"], "empty", "synthetic", [])
    assert client.post("/runs", json={"question_set_id": empty["id"]}).status_code == 409
    too_big = {"retrievers": ["vector", "keyword", "hybrid"], "chunk_sizes": [[400, 0], [800, 0], [1200, 0], [1600, 0]],
               "top_k": [3, 5], "rerank": [False, True], "query_transforms": ["none", "rewrite"]}
    assert client.post("/runs", json={"question_set_id": syn["id"], "grid": too_big}).status_code == 422
