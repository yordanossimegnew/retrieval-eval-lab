"""The three retrievers under test, all over Document QA's tables.

* hybrid  - Document QA's own `DocumentStore.hybrid_search` (vector + full-text, RRF): the
            production retriever, called directly.
* vector  - the vector half of that query on its own (pgvector cosine, HNSW).
* keyword - the full-text half on its own (tsvector, Document QA's `keyword_query`).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

VECTOR_SQL = """
SELECT c.id, c.document_id, c.ord, c.text, c.page_start, c.page_end, d.title, d.source,
       1 - (c.embedding <=> CAST(:qvec AS vector)) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> CAST(:qvec AS vector)
LIMIT :k
"""

KEYWORD_SQL = """
SELECT c.id, c.document_id, c.ord, c.text, c.page_start, c.page_end, d.title, d.source,
       ts_rank_cd(c.tsv, q) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id, to_tsquery('english', :qtext) q
WHERE c.tsv @@ q
ORDER BY ts_rank_cd(c.tsv, q) DESC, c.id
LIMIT :k
"""


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def retrieve(kind: str, store: Any, docqa: Any, query: str, query_vector: list[float] | None, k: int) -> list[dict]:
    if kind == "hybrid":
        hits = store.hybrid_search(query, query_vector, top_k=k, candidates=max(40, k))
    elif kind == "vector":
        with store.engine.connect() as c:
            hits = [dict(r._mapping) for r in c.execute(text(VECTOR_SQL), {"qvec": _vec(query_vector), "k": k})]
    elif kind == "keyword":
        with store.engine.connect() as c:
            hits = [dict(r._mapping) for r in c.execute(text(KEYWORD_SQL), {"qtext": docqa.keyword_query(query), "k": k})]
    else:
        raise ValueError(f"unknown retriever {kind!r}")
    return [{"id": h["id"], "source": h["source"], "title": h["title"], "text": h["text"],
             "page_start": h.get("page_start"), "score": float(h["score"])} for h in hits]
