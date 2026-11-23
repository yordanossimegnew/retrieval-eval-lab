"""Relevance matching and IR metrics.

Ground truth is stored independently of chunking, so the same questions can score every
chunk size fairly. Each question lists relevant *units*:
    {"doc_key": "handbook.pdf", "evidence": "The Pro plan costs $20 per month"}   (synthetic)
    {"doc_key": "beir:4983", "evidence": null}                                     (benchmark)
A retrieved chunk satisfies a unit if it comes from that document and (when evidence is
given) contains the evidence text. Each unit counts once, however many chunks contain it.
"""

from __future__ import annotations

import math
import statistics


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def matched_units(chunk: dict, relevant: list[dict]) -> list[int]:
    text = None
    out = []
    for i, unit in enumerate(relevant):
        if chunk["source"] != unit["doc_key"]:
            continue
        if unit.get("evidence"):
            text = text if text is not None else _norm(chunk["text"])
            if _norm(unit["evidence"]) not in text:
                continue
        out.append(i)
    return out


def score_query(retrieved: list[dict], relevant: list[dict], k: int) -> dict:
    """Recall@k, Precision@k, MRR and nDCG@k (binary gains) for one query."""
    top = retrieved[:k]
    seen: set[int] = set()
    gains, relevant_flags = [], []
    first_hit = None
    for rank, chunk in enumerate(top, 1):
        units = matched_units(chunk, relevant)
        relevant_flags.append(bool(units))
        new = [u for u in units if u not in seen]
        gains.append(1.0 if new else 0.0)  # a chunk earns gain only for units not already found
        seen.update(new)
        if units and first_hit is None:
            first_hit = rank
    n_units = max(1, len(relevant))
    dcg = sum(g / math.log2(r + 1) for r, g in enumerate(gains, 1))
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(k, n_units) + 1))
    return {
        "recall": len(seen) / n_units,
        "precision": sum(relevant_flags) / k,
        "mrr": 1 / first_hit if first_hit else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "hit": 1.0 if seen else 0.0,
    }


def aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0, "queries": 0}
    keys = ("recall", "precision", "mrr", "ndcg", "hit")
    return {**{k: round(statistics.fmean(q[k] for q in per_query), 4) for k in keys}, "queries": len(per_query)}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo), 2)
