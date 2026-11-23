"""IR metrics and relevance matching (pure functions, no database)."""

from __future__ import annotations

import math

import pytest

from src.evals import RetrievalConfig, aggregate, grid, percentile, score_query

REL = [{"doc_key": "a.pdf", "evidence": "The Pro plan costs $20"}]


def chunk(source, text="..."):
    return {"source": source, "text": text}


def test_perfect_and_missed():
    hit = score_query([chunk("a.pdf", "Pricing. The Pro plan costs  $20 per month.")], REL, k=5)
    assert hit == {"recall": 1.0, "precision": 0.2, "mrr": 1.0, "ndcg": 1.0, "hit": 1.0}
    miss = score_query([chunk("a.pdf", "Support details"), chunk("b.md", "The Pro plan costs $20")], REL, k=5)
    assert miss["recall"] == 0 and miss["mrr"] == 0  # right words, wrong document doesn't count


def test_rank_matters():
    ranked_third = [chunk("x"), chunk("y"), chunk("a.pdf", "the pro plan costs $20 a month")]
    m = score_query(ranked_third, REL, k=3)
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["ndcg"] == pytest.approx((1 / math.log2(4)) / 1.0)


def test_duplicate_chunks_count_once_and_document_level_relevance():
    rel = [{"doc_key": "beir:1", "evidence": None}, {"doc_key": "beir:2", "evidence": None}]
    retrieved = [chunk("beir:1"), chunk("beir:1"), chunk("beir:9"), chunk("beir:2")]
    m = score_query(retrieved, rel, k=4)
    assert m["recall"] == 1.0 and m["precision"] == 0.75
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx((1 + 1 / math.log2(5)) / ideal)  # repeat of beir:1 earns no gain


def test_aggregate_and_percentiles():
    assert aggregate([{"recall": 1, "precision": 0.5, "mrr": 1, "ndcg": 1, "hit": 1},
                      {"recall": 0, "precision": 0, "mrr": 0, "ndcg": 0, "hit": 0}])["recall"] == 0.5
    assert percentile([10, 20, 30, 40], 0.5) == 25 and percentile([5], 0.95) == 5


def test_config_grid_and_validation():
    configs = grid(("vector", "hybrid"), ((800, 100), (2400, 300)), (5,), (False, True), ("none",))
    assert len(configs) == 8 and len({c.key for c in configs}) == 8
    assert RetrievalConfig(retriever="hybrid", rerank=True, query_transform="hyde").key == \
        "hybrid · chunk 2400/300 · k=5 · hyde · rerank"
    with pytest.raises(ValueError):
        RetrievalConfig(retriever="bm25")
    with pytest.raises(ValueError):
        RetrievalConfig(chunk_chars=500, chunk_overlap=600)
