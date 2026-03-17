"""Abstract vector backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class VectorBackend(ABC):
    """Abstraction over vector store (pgvector)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend (create tables/indexes)."""
        ...

    @abstractmethod
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
        """Store a fact with embeddings. Returns fact ID."""
        ...

    @abstractmethod
    async def search_by_hq_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[dict, float]]:
        """Search by hypothetical questions embedding. Returns (fact_dict, score)."""
        ...

    @abstractmethod
    async def search_by_fact_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[dict, float]]:
        """Search by fact embedding. Returns (fact_dict, score)."""
        ...

    @abstractmethod
    async def search_by_keywords(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[dict, float]]:
        """Keyword/full-text search. Returns (fact_dict, score)."""
        ...

    @abstractmethod
    async def find_exact_fact(
        self,
        subject: str,
        predicate: str,
        object_val: str,
    ) -> dict | None:
        """Find exact fact match."""
        ...

    @abstractmethod
    async def find_similar_facts(
        self,
        embedding: list[float],
        threshold: float = 0.92,
        limit: int = 3,
    ) -> list[tuple[dict, float]]:
        """Find semantically similar facts."""
        ...

    @abstractmethod
    async def find_facts_by_subject_predicate(
        self,
        subject: str,
        predicate: str,
        only_active: bool = True,
    ) -> list[dict]:
        """Find facts by subject + predicate."""
        ...

    @abstractmethod
    async def invalidate_fact(self, fact_id: str) -> bool:
        """Set valid_until = now on a fact."""
        ...

    @abstractmethod
    async def update_confidence(self, fact_id: str, new_confidence: float) -> bool:
        """Update fact confidence."""
        ...

    @abstractmethod
    async def add_source_to_fact(self, fact_id: str, source: dict) -> bool:
        """Add a source reference to an existing fact."""
        ...

    @abstractmethod
    async def get_active_facts(self, limit: int = 1000) -> list[dict]:
        """Get all active facts for maintenance."""
        ...

    @abstractmethod
    async def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        ...
