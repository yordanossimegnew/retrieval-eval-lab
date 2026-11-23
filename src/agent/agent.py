"""The lab's two LangGraph workflows.

Question generation:   START -> sample -> Send(write) x passages -> save -> END
Evaluation run:        START -> prepare (build indexes) -> Send(evaluate) x configs -> summarize -> END

Configs are evaluated in parallel branches; each branch appends its result through a reducer.
"""

from __future__ import annotations

import operator
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sqlalchemy import insert, update

from ..evals import RetrievalConfig, aggregate, percentile, score_query
from ..evals.datasets import sample_passages, validate_generated
from ..models import Embedder, LabModels, Reranker, Usage
from ..tools import IndexManager, LabDB, retrieve
from ..tools.indexes import estimate_tokens
from ..tools.labdb import question_sets, questions
from ..utils.config import Settings
from ..utils.logger import get_logger

log = get_logger("lab")


@dataclass
class Lab:
    settings: Settings
    db: LabDB
    indexes: IndexManager
    embedder: Embedder
    reranker: Reranker
    models: LabModels
    usage: Usage
    checkpointer: Any = None


# ============================================================================= question generation
class GenState(TypedDict, total=False):
    set_id: str
    corpus_id: str
    n_passages: int
    per_passage: int
    passages: list[dict]
    generated: Annotated[list[dict], operator.add]   # {"passage_index", "items"} per branch


def build_generation_graph(lab: Lab):
    def sample(state: GenState) -> dict:
        docs = lab.db.corpus_documents(state["corpus_id"])
        return {"passages": sample_passages(docs, str(lab.settings.docqa_path), state["n_passages"])}

    def fan_out(state: GenState):
        return [Send("write", {"index": i, "passage": p, "per_passage": state["per_passage"]})
                for i, p in enumerate(state["passages"])] or ["save"]

    def write(task: dict) -> dict:
        p = task["passage"]
        try:
            items = lab.models.write_questions(p["title"], p["passage"], task["per_passage"])
        except Exception as exc:  # one bad passage shouldn't sink the set
            log.warning("question writing failed: %s", exc)
            items = []
        return {"generated": [{"passage_index": task["index"], "items": items}]}

    def save(state: GenState) -> dict:
        seen: set[str] = set()
        kept, dropped = [], []
        for g in sorted(state.get("generated", []), key=lambda g: g["passage_index"]):
            k, d = validate_generated(state["passages"][g["passage_index"]], g["items"], seen)
            kept += k
            dropped += d
        with lab.db.engine.begin() as c:
            if kept:
                c.execute(insert(questions), [{"set_id": state["set_id"], **q} for q in kept])
            c.execute(update(question_sets).where(question_sets.c.id == state["set_id"]).values(meta={
                "status": "ready", "passages": len(state["passages"]), "kept": len(kept), "dropped": len(dropped),
                "dropped_examples": dropped[:20], "usage": lab.usage.snapshot()}))
        return {}

    g = StateGraph(GenState)
    g.add_node("sample", sample)
    g.add_node("write", write)
    g.add_node("save", save)
    g.add_edge(START, "sample")
    g.add_conditional_edges("sample", fan_out, ["write", "save"])
    g.add_edge("write", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=lab.checkpointer)


# ============================================================================= evaluation runs
class RunState(TypedDict, total=False):
    run_id: str
    set_id: str
    corpus_id: str
    configs: list[dict]
    schemas: dict[str, str]          # "chars/overlap" -> index schema
    index_cost: dict
    results: Annotated[list[dict], operator.add]


class _TransformCache:
    """Each (question, mode) is transformed once per run and shared by every config using it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], tuple[str, float, dict]] = {}
        self._pending: dict[tuple[str, str], threading.Event] = {}

    def get(self, key, compute):
        with self._lock:
            if key in self._data:
                return self._data[key]
            event = self._pending.get(key)
            if event is None:
                self._pending[key] = event = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            return self._data[key]
        value = compute()
        with self._lock:
            self._data[key] = value
        event.set()
        return value


def build_eval_graph(lab: Lab):
    s = lab.settings
    cache = _TransformCache()

    def prepare(state: RunState) -> dict:
        lab.db.update_run(state["run_id"], status="running", progress={"stage": "building indexes"})
        schemas, tokens, seconds = {}, 0, 0.0
        for cfg in state["configs"]:
            key = f"{cfg['chunk_chars']}/{cfg['chunk_overlap']}"
            if key in schemas:
                continue
            row = lab.indexes.ensure(state["corpus_id"], cfg["chunk_chars"], cfg["chunk_overlap"],
                                     on_progress=lambda p: lab.db.update_run(state["run_id"], progress=p))
            schemas[key] = row["schema"]
            tokens += row["embed_tokens"]
            seconds += row["build_seconds"]
        lab.db.update_run(state["run_id"], progress={"stage": "evaluating", "indexes": len(schemas)})
        return {"schemas": schemas,
                "index_cost": {"embed_tokens": tokens, "build_seconds": round(seconds, 2),
                               "usd": round(tokens / 1e6 * s.voyage_embed_price, 4)}}

    def fan_out(state: RunState):
        return [Send("evaluate", {"run_id": state["run_id"], "set_id": state["set_id"], "config": cfg,
                                  "schema": state["schemas"][f"{cfg['chunk_chars']}/{cfg['chunk_overlap']}"]})
                for cfg in state["configs"]]

    def evaluate(task: dict) -> dict:
        cfg = RetrievalConfig(**task["config"])
        store = lab.indexes.store(task["schema"])
        questions = lab.db.questions_of(task["set_id"])
        per_query, latencies = [], []
        llm_in = llm_out = embed_tokens = rerank_tokens = 0
        for q in questions:
            def do_transform(question=q["question"]):
                before = lab.usage.snapshot()
                t0 = time.perf_counter()
                text = lab.models.transform_query(question, cfg.query_transform)
                ms = (time.perf_counter() - t0) * 1000
                after = lab.usage.snapshot()
                return text, ms, {"in": after["llm_input"] - before["llm_input"], "out": after["llm_output"] - before["llm_output"]}

            query, transform_ms, tok = (q["question"], 0.0, {"in": 0, "out": 0}) if cfg.query_transform == "none" \
                else cache.get((q["question"], cfg.query_transform), do_transform)
            llm_in += tok["in"]
            llm_out += tok["out"]

            t0 = time.perf_counter()
            vector = None if cfg.retriever == "keyword" else lab.embedder.embed_query(query)
            embed_tokens += 0 if vector is None else estimate_tokens([query])
            depth = max(s.rerank_candidates, cfg.top_k) if cfg.rerank else cfg.top_k
            hits = retrieve(cfg.retriever, store, lab.indexes.docqa, query, vector, depth)
            if cfg.rerank:
                hits = lab.reranker.rerank(q["question"], hits, cfg.top_k)  # rerank against the user's question
                rerank_tokens += estimate_tokens([h["text"] for h in hits]) + estimate_tokens([q["question"]])
            hits = hits[:cfg.top_k]
            latency = transform_ms + (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            metrics = score_query(hits, q["relevant"], cfg.top_k)
            per_query.append({"question_id": q["id"], "query_used": query, "metrics": metrics,
                              "latency_ms": round(latency, 2),
                              "retrieved": [{"id": h["id"], "source": h["source"], "title": h["title"],
                                             "page": h.get("page_start"), "score": round(h.get("rerank_score", h["score"]), 4),
                                             "text": h["text"][:600]} for h in hits]})

        summary = aggregate([p["metrics"] for p in per_query])
        latency = {"p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95),
                   "mean_ms": round(sum(latencies) / max(1, len(latencies)), 2)}
        usd = (llm_in * s.claude_input_price + llm_out * s.claude_output_price
               + embed_tokens * s.voyage_embed_price + rerank_tokens * s.voyage_rerank_price) / 1e6
        cost = {"llm_input_tokens": llm_in, "llm_output_tokens": llm_out, "embed_tokens_est": embed_tokens,
                "rerank_tokens_est": rerank_tokens, "usd": round(usd, 5),
                "usd_per_1k_queries": round(usd / max(1, len(questions)) * 1000, 4)}
        lab.db.save_result(task["run_id"], cfg.key, cfg.to_dict(), summary, latency, cost, per_query)
        return {"results": [{"config_key": cfg.key, "metrics": summary}]}

    def summarize(state: RunState) -> dict:
        best = max(state.get("results", []), key=lambda r: (r["metrics"]["ndcg"], r["metrics"]["recall"]), default=None)
        lab.db.update_run(state["run_id"], status="completed",
                          progress={"stage": "done", "best": best["config_key"] if best else None,
                                    "index_cost": state.get("index_cost")})
        return {}

    g = StateGraph(RunState)
    g.add_node("prepare", prepare)
    g.add_node("evaluate", evaluate)
    g.add_node("summarize", summarize)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", fan_out, ["evaluate"])
    g.add_edge("evaluate", "summarize")
    g.add_edge("summarize", END)
    return g.compile(checkpointer=lab.checkpointer)
