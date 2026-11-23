"""LangGraph checkpointer in PostgreSQL (runs are durable and inspectable)."""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def open_checkpointer(database_url: str) -> PostgresSaver:
    pool = ConnectionPool(database_url.replace("postgresql+psycopg://", "postgresql://"), min_size=1, max_size=10,
                          open=True, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
