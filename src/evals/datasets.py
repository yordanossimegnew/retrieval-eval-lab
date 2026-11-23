"""Corpora and question sets: uploaded files, the SciFact benchmark, and synthetic questions."""

from __future__ import annotations

import csv
import io
import random
import re
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from ..tools.docqa import load_docqa

SCIFACT = {
    "corpus": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels": "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
}


class DatasetError(ValueError):
    pass


# ----------------------------------------------------------------------------- uploaded files
def file_documents(files: list[tuple[str, bytes]], docqa_path: str) -> list[dict]:
    """Validate uploads with Document QA's own loaders; keep raw bytes for re-chunking."""
    dq = load_docqa(docqa_path)
    docs, seen = [], set()
    for name, data in files:
        if name in seen:
            raise DatasetError(f"Duplicate file name: {name}")
        seen.add(name)
        try:
            loaded = dq.load_file(name, data)
        except ValueError as exc:
            raise DatasetError(f"{name}: {exc}") from exc
        docs.append({"doc_key": name, "title": loaded.title, "filename": name, "raw": data, "text": None})
    if not docs:
        raise DatasetError("No files provided.")
    return docs


# ----------------------------------------------------------------------------- SciFact (BEIR)
def _download(url: str, cache: Path, client: httpx.Client) -> bytes:
    path = cache / url.rsplit("/", 1)[-1]
    if path.exists():
        return path.read_bytes()
    response = client.get(url, follow_redirects=True, timeout=120)
    if response.status_code != 200:
        raise DatasetError(f"Download failed ({response.status_code}): {url}")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def scifact_subset(cache: Path, max_queries: int = 100, max_docs: int = 800, seed: int = 7,
                   client: httpx.Client | None = None) -> tuple[list[dict], list[dict]]:
    """Test queries with their relevant abstracts, plus random distractor abstracts."""
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": "retrieval-eval-lab/1.0"})
    try:
        corpus = pq.read_table(io.BytesIO(_download(SCIFACT["corpus"], cache, client))).to_pylist()
        queries = {str(q["_id"]): q["text"] for q in pq.read_table(io.BytesIO(_download(SCIFACT["queries"], cache, client))).to_pylist()}
        qrels_raw = _download(SCIFACT["qrels"], cache, client).decode()
    finally:
        if own:
            client.close()

    qrels: dict[str, list[str]] = {}
    for row in csv.DictReader(io.StringIO(qrels_raw), delimiter="\t"):
        if int(row["score"]) > 0:
            qrels.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
    rng = random.Random(seed)
    query_ids = sorted(q for q in qrels if q in queries)
    rng.shuffle(query_ids)
    query_ids = query_ids[:max_queries]

    by_id = {str(d["_id"]): d for d in corpus}
    needed = {doc for q in query_ids for doc in qrels[q] if doc in by_id}
    distractors = [i for i in by_id if i not in needed]
    rng.shuffle(distractors)
    chosen = list(needed) + distractors[:max(0, max_docs - len(needed))]

    docs = [{"doc_key": f"beir:{i}", "title": by_id[i]["title"] or f"Abstract {i}", "filename": None, "raw": None,
             "text": f"{by_id[i]['title']}\n\n{by_id[i]['text']}".strip()} for i in chosen]
    items = [{"question": queries[q], "relevant": [{"doc_key": f"beir:{d}", "evidence": None} for d in qrels[q] if d in by_id],
              "meta": {"beir_query_id": q}} for q in query_ids]
    return docs, items


# ----------------------------------------------------------------------------- synthetic questions
def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def sample_passages(db_docs: list[dict], docqa_path: str, n: int, seed: int = 7, chars: int = 1200) -> list[dict]:
    """Spread n passages across documents (round robin), skipping fragments too short to ask about."""
    dq = load_docqa(docqa_path)
    per_doc = []
    for row in db_docs:
        if row["raw"] is not None:
            sections = dq.load_file(row["filename"], bytes(row["raw"])).sections
        else:
            sections = [dq.Section(text=row["text"])]
        chunks = [c for c in dq.chunk_sections(sections, chars, 0) if len(c.text) >= 200]
        random.Random(seed).shuffle(chunks)
        per_doc.append([{"doc_key": row["doc_key"], "title": row["title"], "passage": c.text} for c in chunks])
    out = []
    while len(out) < n and any(per_doc):
        for lst in per_doc:
            if lst and len(out) < n:
                out.append(lst.pop())
    return out


def validate_generated(passage: dict, generated: list[dict], seen: set[str]) -> tuple[list[dict], list[dict]]:
    """Keep questions whose evidence is an exact quote from the passage; drop vague or duplicate ones."""
    kept, dropped = [], []
    text = _norm(passage["passage"])
    for g in generated:
        q, ev = g.get("question", "").strip(), g.get("evidence", "").strip().strip('"“”')
        key = _norm(q)
        reason = None
        if len(q) < 12 or not q.endswith("?"):
            reason = "not a well-formed question"
        elif len(ev) < 12:
            reason = "evidence too short"
        elif _norm(ev) not in text:
            reason = "evidence is not an exact quote from the passage"
        elif key in seen:
            reason = "duplicate question"
        elif len(set(re.findall(r"\w+", key)) & set(re.findall(r"\w+", _norm(ev)))) / max(1, len(set(re.findall(r"\w+", key)))) > 0.9:
            reason = "question copies the passage almost verbatim"
        if reason:
            dropped.append({"question": q, "reason": reason})
            continue
        seen.add(key)
        kept.append({"question": q, "relevant": [{"doc_key": passage["doc_key"], "evidence": ev}],
                     "meta": {"title": passage["title"]}})
    return kept, dropped
