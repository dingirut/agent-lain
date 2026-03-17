"""Data models for the semantic memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DocType(str, Enum):
    """Document types in memory."""
    MEMORY = "memory"
    GUIDE = "guide"
    ANSWER = "answer"
    CODE = "code"
    CONVERSATION = "conversation"


class ExtractionType(str, Enum):
    """How a fact was extracted."""
    AUTO = "auto"              # LLM-extracted from conversation
    USER_STATED = "user_stated"  # User explicitly said "remember this"
    INFERENCE = "inference"    # Agent inferred from context
    TOOL_RESULT = "tool_result"  # Extracted from tool execution


class FactCategory(str, Enum):
    """Fact categories for decay rules and filtering."""
    PERSONAL = "personal"
    PROJECT = "project"
    TECHNICAL = "technical"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    ORGANIZATION = "organization"
    PERSON = "person"
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    GEOPOLITICS = "geopolitics"
    SCHEME = "scheme"
    OTHER = "other"


@dataclass
class Entity:
    """A named entity in the knowledge graph."""
    id: str = ""
    name: str = ""
    entity_type: str = ""  # person, project, tool, technology, etc.
    aliases: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Fact:
    """A single fact (triple) in memory."""
    id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""

    # Confidence & provenance
    confidence: float = 1.0
    original_confidence: float = 1.0
    extraction_type: ExtractionType = ExtractionType.AUTO
    evidence: str | None = None  # exact quote from source text
    source_session: str | None = None

    # Metadata (enriched)
    category: FactCategory = FactCategory.OTHER
    hypothetical_questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Embeddings (stored in pgvector, not in dataclass)
    # fact_embedding: vector  -- in DB only
    # hq_embedding: vector    -- in DB only

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None  # None = still valid
    last_confirmed: datetime = field(default_factory=datetime.now)
    time_context: str | None = None  # e.g. "2024", "since 2022", "Feb 2025"

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    @property
    def fact_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    @property
    def hq_text(self) -> str:
        return " | ".join(self.hypothetical_questions) if self.hypothetical_questions else self.fact_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "extraction_type": self.extraction_type.value,
            "category": self.category.value,
            "hypothetical_questions": self.hypothetical_questions,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "last_confirmed": self.last_confirmed.isoformat(),
            "evidence": self.evidence,
            "time_context": self.time_context,
        }


@dataclass
class SearchResult:
    """A scored fact from retrieval."""
    fact: Fact
    score: float
    sources: list[str] = field(default_factory=list)  # which search methods found it
    issues: list[str] = field(default_factory=list)    # freshness/confidence warnings
    note: str | None = None

    def format_for_llm(self, index: int) -> str:
        """Format for injection into LLM context."""
        f = self.fact
        parts = [f"### Memory Fact {index + 1} (relevance: {self.score:.0%})"]

        if f.category != FactCategory.OTHER:
            parts.append(f"**Category:** {f.category.value}")
        parts.append(f"**{f.subject}** {f.predicate} **{f.object}**")
        if f.evidence:
            parts.append(f"_Evidence: {f.evidence}_")
        if self.issues:
            parts.append(f"_Issues: {', '.join(self.issues)}_")
        if self.note:
            parts.append(f"_Note: {self.note}_")

        return "\n".join(parts)


@dataclass
class IngestionReport:
    """Report from the ingestion pipeline."""
    extracted: int = 0
    stored: int = 0
    filtered_noise: int = 0
    deduplicated: int = 0
    hallucinated: int = 0
    superseded: int = 0
    errors: list[str] = field(default_factory=list)
    hallucination_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"Extracted: {self.extracted}"]
        if self.stored:
            parts.append(f"stored: {self.stored}")
        if self.filtered_noise:
            parts.append(f"noise filtered: {self.filtered_noise}")
        if self.deduplicated:
            parts.append(f"duplicates: {self.deduplicated}")
        if self.hallucinated:
            parts.append(f"hallucinations blocked: {self.hallucinated}")
        if self.superseded:
            parts.append(f"superseded: {self.superseded}")
        return " | ".join(parts)


@dataclass
class GroundingResult:
    """Result of grounding check."""
    grounded: bool = False
    evidence: str | None = None
    confidence: float = 0.5


@dataclass
class DuplicateResult:
    """Result of duplicate check."""
    is_duplicate: bool = False
    existing_fact_id: str | None = None
    action: str = ""  # "update_source", "merge_or_skip"
    similarity: float = 0.0


@dataclass
class CoherenceResult:
    """Result of coherence check."""
    coherent: bool = True
    conflict_type: str = ""  # "supersedes"
    old_fact_id: str | None = None
    old_value: str = ""
    new_value: str = ""
    action: str = ""  # "invalidate_old"
    note: str = ""


@dataclass
class EnrichedMetadata:
    """Enriched metadata for a fact."""
    hypothetical_questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    category: str = "other"


@dataclass
class Source:
    """Source information for provenance tracking."""
    session_key: str = ""
    channel: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    message_index: int = 0  # position in conversation
