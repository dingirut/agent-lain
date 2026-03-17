"""Hybrid retriever: HQ vector + fact vector + keyword + graph."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.embeddings import EmbeddingProvider
from ragnarbot.agent.memory.models import (
    ExtractionType,
    Fact,
    FactCategory,
    SearchResult,
)
from ragnarbot.agent.memory.validation import RecallValidator


def _dict_to_fact(d: dict) -> Fact:
    """Convert a DB row dict to a Fact dataclass."""
    return Fact(
        id=d.get("id", ""),
        subject=d.get("subject", ""),
        predicate=d.get("predicate", ""),
        object=d.get("object", ""),
        confidence=float(d.get("confidence", 1.0)),
        original_confidence=float(d.get("original_confidence", 1.0)),
        extraction_type=ExtractionType(d.get("extraction_type", "auto")),
        evidence=d.get("evidence"),
        source_session=d.get("source_session"),
        category=FactCategory(d.get("category", "other")),
        hypothetical_questions=d.get("hypothetical_questions") or [],
        keywords=d.get("keywords") or [],
        created_at=datetime.fromisoformat(d["created_at"]) if isinstance(d.get("created_at"), str) else (d.get("created_at") or datetime.now()),
        valid_from=datetime.fromisoformat(d["valid_from"]) if isinstance(d.get("valid_from"), str) else (d.get("valid_from") or datetime.now()),
        valid_until=datetime.fromisoformat(d["valid_until"]) if isinstance(d.get("valid_until"), str) and d.get("valid_until") else d.get("valid_until"),
        last_confirmed=datetime.fromisoformat(d["last_confirmed"]) if isinstance(d.get("last_confirmed"), str) else (d.get("last_confirmed") or datetime.now()),
    )


class HybridRetriever:
    """3-stage retrieval: gather → merge → validate.

    Search channels:
    1. HQ vector search (best for question matching)
    2. Fact vector search (entity lookup)
    3. Keyword search (exact name matches)
    4. Graph traversal (related entities via Neo4j)
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        backend: Any,  # PgvectorBackend
        graph: Any | None = None,  # Neo4jBackend
        recall_validator: RecallValidator | None = None,
    ):
        self.embedder = embedder
        self.backend = backend
        self.graph = graph
        self.validator = recall_validator or RecallValidator()

    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.1,
        mode: str = "normal",
    ) -> list[SearchResult]:
        """Hybrid search across all channels."""
        query_embedding = await self.embedder.embed_query(query)

        candidates: dict[str, dict] = {}

        # Channel 1: HQ vector search (highest weight)
        try:
            hq_results = await self.backend.search_by_hq_embedding(
                query_embedding, limit=limit * 2, score_threshold=score_threshold,
            )
            for row, score in hq_results:
                fid = row["id"]
                candidates[fid] = {
                    "row": row, "hq_score": score, "sources": ["hq_vector"],
                }
        except Exception as e:
            logger.warning(f"HQ vector search failed: {e}")

        # Channel 2: Fact vector search
        try:
            fact_results = await self.backend.search_by_fact_embedding(
                query_embedding, limit=limit * 2, score_threshold=score_threshold,
            )
            for row, score in fact_results:
                fid = row["id"]
                if fid in candidates:
                    candidates[fid]["fact_score"] = score
                    candidates[fid]["sources"].append("fact_vector")
                else:
                    candidates[fid] = {
                        "row": row, "fact_score": score, "sources": ["fact_vector"],
                    }
        except Exception as e:
            logger.warning(f"Fact vector search failed: {e}")

        # Channel 3: Keyword search
        try:
            kw_results = await self.backend.search_by_keywords(query, limit=limit)
            for row, score in kw_results:
                fid = row["id"]
                if fid in candidates:
                    candidates[fid]["keyword_match"] = True
                    candidates[fid]["sources"].append("keyword")
                else:
                    candidates[fid] = {
                        "row": row, "keyword_match": True, "sources": ["keyword"],
                    }
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")

        # Channel 4: Graph search
        if self.graph:
            try:
                entity_name = await self.graph.resolve_entity_from_query(query)
                if entity_name:
                    graph_results = await self.graph.get_related_facts(
                        entity_name, depth=2, limit=limit,
                    )
                    for gr in graph_results:
                        fid = gr.get("fact_id", "")
                        if not fid:
                            continue
                        if fid in candidates:
                            candidates[fid]["graph_depth"] = gr.get("depth", 1)
                            candidates[fid]["sources"].append("graph")
                        else:
                            # Graph-only results need their row data
                            # We'd need to fetch from DB — skip for now, just boost existing
                            pass
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")

        # Merge scores
        scored_results = []
        for cid, data in candidates.items():
            score = 0.0
            # HQ vector: highest weight (question-to-question matching)
            score += data.get("hq_score", 0) * 0.4
            # Fact vector
            score += data.get("fact_score", 0) * 0.25
            # Keyword exact match bonus
            if data.get("keyword_match"):
                score += 0.2
            # Graph proximity
            if "graph_depth" in data:
                score += 0.15 / data["graph_depth"]
            # Multi-source bonus
            source_count = len(data["sources"])
            if source_count > 1:
                score *= (1 + 0.1 * (source_count - 1))

            fact = _dict_to_fact(data["row"])
            scored_results.append(SearchResult(
                fact=fact,
                score=score,
                sources=data["sources"],
            ))

        # Sort by score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # Recall validation (freshness, confidence, coherence)
        validated = await self.validator.validate_recall(
            query, scored_results[:limit * 2], mode=mode,
        )

        return validated[:limit]

    async def search_for_context(
        self,
        query: str,
        max_results: int = 3,
        max_chars: int = 2000,
    ) -> str:
        """Search and format results for system prompt injection."""
        results = await self.search(query, limit=max_results)
        if not results:
            return ""

        parts = []
        total_chars = 0
        for i, sr in enumerate(results):
            formatted = sr.format_for_llm(i)
            if total_chars + len(formatted) > max_chars:
                break
            parts.append(formatted)
            total_chars += len(formatted)

        return "\n\n".join(parts)
