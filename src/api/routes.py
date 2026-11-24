"""FastAPI app: corpora, question sets, experiment runs and results."""

from __future__ import annotations

import threading
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from ..agent import Lab, build_eval_graph, build_generation_graph, open_checkpointer
from ..evals import grid
from ..evals.datasets import DatasetError, file_documents, scifact_subset
from ..models import AnthropicLabModels, Usage, VoyageReranker, make_embedder
from ..tools import DocqaNotFound, IndexManager, LabDB, load_docqa
from ..utils.config import PROJECT_ROOT, Settings, load_settings
from ..utils.logger import get_logger
from .schemas import BenchmarkRequest, HealthResponse, RunRequest, SyntheticRequest

log = get_logger("api")


def create_app(settings: Settings | None = None, lab_factory: Callable[[LabDB], Lab] | None = None,
               scifact_loader: Callable[..., tuple[list, list]] | None = None, run_inline: bool = False) -> FastAPI:
    settings = settings or load_settings()
    db = LabDB(settings.database_url)
    db.init()
    state: dict[str, Any] = {}

    def lab() -> Lab:
        if "lab" not in state:
            if lab_factory:
                state["lab"] = lab_factory(db)
            else:
                try:
                    load_docqa(str(settings.docqa_path))
                    usage = Usage()
                    embedder = make_embedder(settings)
                except DocqaNotFound as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(status_code=503, detail=f"Models unavailable: {exc}. Set VOYAGE_API_KEY.") from exc
                state["lab"] = Lab(settings, db, IndexManager(db, settings.docqa_path, embedder, f"voyage/{settings.voyage_model}",
                                                              settings.embedding_dim),
                                   embedder, VoyageReranker(settings, usage), AnthropicLabModels(settings, usage), usage,
                                   open_checkpointer(settings.database_url))
        return state["lab"]

    def background(target, *args):
        if run_inline:  # tests
            target(*args)
        else:
            threading.Thread(target=target, args=args, daemon=True).start()

    if lab_factory:
        lab()  # injected labs are cheap to build; build now so /health reports them
    app = FastAPI(title="Retrieval Evaluation Lab", version="1.0.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # Report the models actually in use once the lab exists (they can be stand-ins in tests/demos).
        active = state.get("lab")
        return HealthResponse(
            model=getattr(active.models, "label", settings.model) if active else settings.model,
            embeddings=active.indexes.embed_model if active else f"voyage/{settings.voyage_model}",
            reranker=getattr(active.reranker, "label", f"voyage/{settings.rerank_model}") if active else f"voyage/{settings.rerank_model}",
            docqa_path=str(settings.docqa_path), database=settings.database_url.split("@")[-1])

    # ------------------------------------------------------------------ corpora
    @app.post("/corpora", status_code=201)
    async def upload_corpus(name: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
        payload = [(f.filename or "file", await f.read()) for f in files]
        try:
            docs = file_documents(payload, str(settings.docqa_path))
        except (DatasetError, DocqaNotFound) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return db.add_corpus(name, "files", docs)

    @app.post("/corpora/benchmark", status_code=201)
    def add_benchmark(request: BenchmarkRequest) -> dict:
        loader = scifact_loader or scifact_subset
        try:
            docs, items = loader(PROJECT_ROOT / "data" / "benchmarks" / "scifact", max_queries=request.max_queries,
                                 max_docs=request.max_docs)
        except DatasetError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        corpus = db.add_corpus(f"SciFact ({len(docs)} abstracts)", "beir",
                               docs, {"dataset": "scifact", "source": "BEIR / Hugging Face"})
        qset = db.add_question_set(corpus["id"], f"SciFact test queries ({len(items)})", "benchmark", items,
                                   {"status": "ready", "dataset": "scifact"})
        return {"corpus": corpus, "question_set": qset}

    @app.get("/corpora")
    def list_corpora() -> list[dict]:
        return db.list_corpora()

    # ------------------------------------------------------------------ question sets
    @app.post("/question-sets/synthetic", status_code=202)
    def synthetic(request: SyntheticRequest) -> dict:
        if db.corpus(request.corpus_id) is None:
            raise HTTPException(status_code=404, detail="Corpus not found")
        qset = db.add_question_set(request.corpus_id, request.name, "synthetic", [], {"status": "generating"})

        def job():
            try:
                graph = build_generation_graph(lab())
                graph.invoke({"set_id": qset["id"], "corpus_id": request.corpus_id, "n_passages": request.passages,
                              "per_passage": request.per_passage}, {"configurable": {"thread_id": f"gen-{qset['id']}"}})
            except Exception as exc:
                log.exception("question generation failed")
                db.set_question_set_meta(qset["id"], {"status": "failed", "error": str(exc)[:500]})
        background(job)
        return db.question_set(qset["id"])

    @app.get("/question-sets")
    def list_question_sets() -> list[dict]:
        return db.list_question_sets()

    @app.get("/question-sets/{set_id}")
    def question_set(set_id: str) -> dict:
        qs = db.question_set(set_id)
        if qs is None:
            raise HTTPException(status_code=404, detail="Question set not found")
        return {**qs, "items": db.questions_of(set_id)}

    # ------------------------------------------------------------------ runs
    @app.post("/runs", status_code=202)
    def start_run(request: RunRequest) -> dict:
        qs = db.question_set(request.question_set_id)
        if qs is None:
            raise HTTPException(status_code=404, detail="Question set not found")
        if qs["questions"] == 0:
            raise HTTPException(status_code=409, detail="The question set has no questions (yet).")
        g = request.grid
        try:
            configs = grid(g.retrievers, [tuple(x) for x in g.chunk_sizes], g.top_k, g.rerank, g.query_transforms)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(configs) > 48:
            raise HTTPException(status_code=422, detail=f"{len(configs)} configurations; keep it to 48 or fewer.")
        run = db.create_run(qs["id"], request.name, [c.to_dict() for c in configs])

        def job():
            try:
                graph = build_eval_graph(lab())
                graph.invoke({"run_id": run["id"], "set_id": qs["id"], "corpus_id": qs["corpus_id"],
                              "configs": [c.to_dict() for c in configs]},
                             {"configurable": {"thread_id": f"run-{run['id']}"},
                              "max_concurrency": settings.max_parallel_configs})
            except Exception as exc:
                log.exception("run failed")
                db.update_run(run["id"], status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        background(job)
        return db.run(run["id"])

    @app.get("/runs")
    def list_runs() -> list[dict]:
        return db.list_runs()

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        run = db.run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run["progress"] = {**run["progress"], "done": len(run["results"]), "total": len(run["configs"])}
        run["results"].sort(key=lambda r: (-r["metrics"]["ndcg"], -r["metrics"]["recall"], r["latency"]["p50_ms"]))
        return run

    @app.get("/runs/{run_id}/queries")
    def run_queries(run_id: str, config_key: str = Query(...), only_misses: bool = False) -> list[dict]:
        rows = db.run_queries_of(run_id, config_key)
        return [r for r in rows if not only_misses or r["metrics"]["recall"] < 1]

    @app.get("/runs/{run_id}/compare")
    def compare(run_id: str, a: str = Query(...), b: str = Query(...)) -> dict:
        """Per-question differences between two configs (which questions one finds and the other misses)."""
        qa = {r["question_id"]: r for r in db.run_queries_of(run_id, a)}
        qb = {r["question_id"]: r for r in db.run_queries_of(run_id, b)}
        wins_a = [qa[i]["question"] for i in qa if i in qb and qa[i]["metrics"]["hit"] > qb[i]["metrics"]["hit"]]
        wins_b = [qb[i]["question"] for i in qb if i in qa and qb[i]["metrics"]["hit"] > qa[i]["metrics"]["hit"]]
        return {"a": a, "b": b, "only_a_found": wins_a, "only_b_found": wins_b}

    return app
