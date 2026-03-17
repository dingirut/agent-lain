"""PostgreSQL + pgvector backend for semantic memory."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger

from .base import VectorBackend

# fmt: off
INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,

    -- confidence & provenance
    confidence FLOAT NOT NULL DEFAULT 1.0,
    original_confidence FLOAT NOT NULL DEFAULT 1.0,
    extraction_type TEXT NOT NULL DEFAULT 'auto',
    evidence TEXT,
    source_session TEXT,

    -- metadata
    category TEXT NOT NULL DEFAULT 'other',
    hypothetical_questions TEXT[],
    keywords TEXT[],

    -- embeddings
    fact_embedding vector({dimensions}),
    hq_embedding vector({dimensions}),

    -- temporal
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    last_confirmed TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- sources (JSONB array of source objects)
    sources JSONB NOT NULL DEFAULT '[]'
);

-- HNSW indexes for fast cosine similarity
CREATE INDEX IF NOT EXISTS idx_facts_fact_embedding
    ON facts USING hnsw (fact_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_facts_hq_embedding
    ON facts USING hnsw (hq_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Keyword search
CREATE INDEX IF NOT EXISTS idx_facts_keywords
    ON facts USING gin (keywords);

-- Subject + predicate for coherence checks
CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate
    ON facts (subject, predicate);

-- Category filter
CREATE INDEX IF NOT EXISTS idx_facts_category
    ON facts (category);

-- Active facts (valid_until IS NULL)
CREATE INDEX IF NOT EXISTS idx_facts_active
    ON facts (valid_until) WHERE valid_until IS NULL;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_facts_fts
    ON facts USING gin (
        to_tsvector('english',
            coalesce(subject, '') || ' ' ||
            coalesce(predicate, '') || ' ' ||
            coalesce(object, '')
        )
    );
"""
# fmt: on


def _vec_str(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _row_to_dict(row) -> dict:
    """Convert asyncpg Record to fact dict."""
    d = dict(row)
    # Convert UUID to string
    if "id" in d:
        d["id"] = str(d["id"])
    # Convert datetime objects
    for k in ("created_at", "valid_from", "valid_until", "last_confirmed"):
        if k in d and d[k] is not None:
            d[k] = d[k].isoformat()
    # Convert sources from JSON
    if "sources" in d and isinstance(d["sources"], str):
        d["sources"] = json.loads(d["sources"])
    return d


class PgvectorBackend(VectorBackend):
    """PostgreSQL + pgvector implementation."""

    def __init__(self, database_url: str, dimensions: int = 768):
        self.database_url = database_url
        self.dimensions = dimensions
        self._pool = None

    async def initialize(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=1, max_size=5,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(INIT_SQL.format(dimensions=self.dimensions))
        logger.info(f"pgvector backend initialized (dimensions={self.dimensions})")

    async def store_fact(
        self,
        fact_id: str,
        subject: str,
        predicate: str,
        object_val: str,
        fact_embedding: list[float],
        hq_embedding: list[float] | None,
        metadata: dict[str, str],
    ) -> str:
        if not fact_id:
            fact_id = str(uuid.uuid4())

        sql = """
        INSERT INTO facts (
            id, subject, predicate, object,
            confidence, original_confidence, extraction_type,
            evidence, source_session, category,
            hypothetical_questions, keywords,
            fact_embedding, hq_embedding, sources
        ) VALUES (
            $1, $2, $3, $4,
            $5, $6, $7,
            $8, $9, $10,
            $11, $12,
            $13::vector, $14::vector, $15::jsonb
        )
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                sql,
                uuid.UUID(fact_id),
                subject,
                predicate,
                object_val,
                float(metadata.get("confidence", 1.0)),
                float(metadata.get("original_confidence", 1.0)),
                metadata.get("extraction_type", "auto"),
                metadata.get("evidence"),
                metadata.get("source_session"),
                metadata.get("category", "other"),
                metadata.get("hypothetical_questions", "").split("|") if metadata.get("hypothetical_questions") else [],
                metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                _vec_str(fact_embedding),
                _vec_str(hq_embedding) if hq_embedding else None,
                metadata.get("sources", "[]"),
            )
        return fact_id

    async def search_by_hq_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[dict, float]]:
        return await self._vector_search(
            "hq_embedding", query_embedding, limit, score_threshold, filters,
        )

    async def search_by_fact_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[dict, float]]:
        return await self._vector_search(
            "fact_embedding", query_embedding, limit, score_threshold, filters,
        )

    async def _vector_search(
        self,
        column: str,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
        filters: Optional[dict[str, str]],
    ) -> list[tuple[dict, float]]:
        where_clauses = [f"1 - ({column} <=> $1::vector) >= $2"]
        params: list = [_vec_str(query_embedding), score_threshold]
        idx = 3

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"{key} = ${idx}")
                params.append(value)
                idx += 1

        where = " AND ".join(where_clauses)
        sql = f"""
        SELECT *, 1 - ({column} <=> $1::vector) AS score
        FROM facts
        WHERE valid_until IS NULL AND {where}
        ORDER BY {column} <=> $1::vector
        LIMIT ${idx}
        """
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [(_row_to_dict(row), float(row["score"])) for row in rows]

    async def search_by_keywords(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[dict, float]]:
        sql = """
        SELECT *,
            ts_rank(
                to_tsvector('english', coalesce(subject,'') || ' ' || coalesce(predicate,'') || ' ' || coalesce(object,'')),
                plainto_tsquery('english', $1)
            ) AS score
        FROM facts
        WHERE valid_until IS NULL
            AND to_tsvector('english', coalesce(subject,'') || ' ' || coalesce(predicate,'') || ' ' || coalesce(object,''))
                @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, query, limit)
        return [(_row_to_dict(row), float(row["score"])) for row in rows]

    async def find_exact_fact(
        self,
        subject: str,
        predicate: str,
        object_val: str,
    ) -> dict | None:
        sql = """
        SELECT * FROM facts
        WHERE subject = $1 AND predicate = $2 AND object = $3
            AND valid_until IS NULL
        LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, subject, predicate, object_val)
        return _row_to_dict(row) if row else None

    async def find_similar_facts(
        self,
        embedding: list[float],
        threshold: float = 0.92,
        limit: int = 3,
    ) -> list[tuple[dict, float]]:
        sql = """
        SELECT *, 1 - (fact_embedding <=> $1::vector) AS score
        FROM facts
        WHERE valid_until IS NULL
            AND 1 - (fact_embedding <=> $1::vector) >= $2
        ORDER BY fact_embedding <=> $1::vector
        LIMIT $3
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, _vec_str(embedding), threshold, limit)
        return [(_row_to_dict(row), float(row["score"])) for row in rows]

    async def find_facts_by_subject_predicate(
        self,
        subject: str,
        predicate: str,
        only_active: bool = True,
    ) -> list[dict]:
        if only_active:
            sql = "SELECT * FROM facts WHERE subject = $1 AND predicate = $2 AND valid_until IS NULL"
        else:
            sql = "SELECT * FROM facts WHERE subject = $1 AND predicate = $2"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, subject, predicate)
        return [_row_to_dict(row) for row in rows]

    async def invalidate_fact(self, fact_id: str) -> bool:
        sql = "UPDATE facts SET valid_until = NOW() WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, uuid.UUID(fact_id))
        return "UPDATE 1" in result

    async def update_confidence(self, fact_id: str, new_confidence: float) -> bool:
        sql = "UPDATE facts SET confidence = $2 WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, uuid.UUID(fact_id), new_confidence)
        return "UPDATE 1" in result

    async def add_source_to_fact(self, fact_id: str, source: dict) -> bool:
        sql = """
        UPDATE facts
        SET sources = sources || $2::jsonb,
            last_confirmed = NOW()
        WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                sql, uuid.UUID(fact_id), json.dumps([source]),
            )
        return "UPDATE 1" in result

    async def get_active_facts(self, limit: int = 1000) -> list[dict]:
        sql = "SELECT * FROM facts WHERE valid_until IS NULL ORDER BY last_confirmed ASC LIMIT $1"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, limit)
        return [_row_to_dict(row) for row in rows]

    async def delete_fact(self, fact_id: str) -> bool:
        sql = "DELETE FROM facts WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, uuid.UUID(fact_id))
        return "DELETE 1" in result

    async def purge_all(self) -> int:
        """Delete ALL facts. Returns count of deleted rows."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM facts")
        # result like "DELETE 42"
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
