"""Validation gates for memory ingestion and recall.

Ingestion: Importance → Duplicate → Grounding → Coherence
Recall: Freshness → Confidence → Coherence in context
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.models import (
    CoherenceResult,
    DuplicateResult,
    ExtractionType,
    Fact,
    GroundingResult,
    SearchResult,
)


# ── Ingestion Gate 1: Importance Filter (free, no LLM) ──────────

ALWAYS_IMPORTANT = {
    # Personal / project
    "works_on", "works_at", "uses", "prefers",
    "decided", "wants", "lives_in", "knows",
    "current_role", "created", "deployed_to", "configured",
    "manages", "owns", "maintains", "built",
    "speaks", "timezone", "primary_language",
    "has_skill", "member_of", "reports_to",
    # Organizational / structural
    "controls", "finances", "is_member_of", "leads",
    "appointed_to", "founded", "is_beneficiary_of",
    "funded_by", "has_offshore_in", "contracted_with",
    "is_subsidiary_of",
    # Technology
    "develops", "supplies", "is_type_of", "is_class_of",
    "used_for", "deployed_in", "evolved_from", "replaces",
    "built_on", "has_countermeasure", "vulnerable_to",
    "has_range", "has_precision", "manufactured_in",
    # Conceptual
    "is_doctrine", "is_methodology", "is_concept", "is_technique",
    "consists_of", "implemented_via", "contradicts", "complements",
    "adopted_in", "emerged_as_response_to", "is_variant_of",
    # Geopolitical
    "allied_with", "opposes", "influences", "depends_on",
    "under_sanctions", "supplies_weapons_to", "funds_regime",
}

CONDITIONAL = {
    "related_to", "interested_in", "learning",
}

NOISE = {
    # Greetings / meta
    "greeted", "said_bye", "asked_about_weather",
    "thanked", "acknowledged", "confirmed", "agreed",
    # Transient task actions (conversation flow, not facts)
    "requested", "asked", "asked_for", "asked_about",
    "approach", "approached", "translated", "generated",
    "summarized", "explained", "analyzed", "reviewed",
    "will_do", "started", "finished", "completed",
    # Time estimates / progress
    "estimated_duration", "eta", "progress", "status",
    "took_time", "duration",
    # Meta about the conversation itself
    "discussed", "mentioned", "talked_about", "responded",
    "clarified", "elaborated", "suggested",
}

# Subjects that should never produce stored facts
NOISE_SUBJECTS = {
    "assistant", "ai", "bot", "ragnarbot", "claude",
    "system", "conversation", "chat", "session",
    "translation", "translation_task", "task", "request",
}


def check_importance(fact: Fact) -> bool:
    """Filter noise facts. Returns True if fact should be kept."""
    predicate = fact.predicate.lower().strip()
    confidence = fact.confidence
    subject_lower = fact.subject.lower().strip()

    # Block facts about the assistant or conversation meta-entities
    if subject_lower in NOISE_SUBJECTS:
        return False

    # Block noise predicates
    if predicate in NOISE:
        return False

    # Always keep important predicates
    if predicate in ALWAYS_IMPORTANT:
        return True

    # Conditional predicates need high confidence
    if predicate in CONDITIONAL:
        return confidence > 0.8

    # Unknown predicate — require high confidence AND non-trivial subject
    # This catches novel predicates that the LLM invents
    if confidence < 0.75:
        return False

    # Reject very short or generic subjects/objects
    if len(fact.subject.strip()) < 2 or len(fact.object.strip()) < 2:
        return False

    return True


# ── Ingestion Gate 2: Duplicate Check (DB query) ────────────────

class DuplicateChecker:
    """Check if a fact already exists in memory."""

    def __init__(self, backend: Any, embedder: Any):
        self.backend = backend
        self.embedder = embedder

    async def check(self, fact: Fact) -> DuplicateResult:
        # 1. Exact match
        existing = await self.backend.find_exact_fact(
            subject=fact.subject,
            predicate=fact.predicate,
            object_val=fact.object,
        )
        if existing:
            return DuplicateResult(
                is_duplicate=True,
                existing_fact_id=existing["id"],
                action="update_source",
            )

        # 2. Semantic similarity
        fact_embedding = await self.embedder.embed_query(fact.fact_text)
        similar = await self.backend.find_similar_facts(
            fact_embedding, threshold=0.92, limit=1,
        )
        if similar:
            row, score = similar[0]
            return DuplicateResult(
                is_duplicate=True,
                existing_fact_id=row["id"],
                action="merge_or_skip",
                similarity=score,
            )

        return DuplicateResult(is_duplicate=False)


# ── Ingestion Gate 3: Grounding Check (LLM call) ────────────────

GROUNDING_PROMPT = """Given the original text and an extracted fact,
determine if the fact is GROUNDED in the text.

Original text:
{source_text}

Extracted fact:
Subject: {subject}
Predicate: {predicate}
Object: {object}

Answer with JSON:
{{
  "grounded": true/false,
  "evidence": "exact quote from text that supports this fact, or null",
  "confidence_adjustment": 0.0-1.0
}}

Rules:
- GROUNDED = the fact can be directly inferred from the text
- NOT GROUNDED = the fact adds information not present in the text
- Slight paraphrasing is OK, but inventing details is NOT"""


class GroundingChecker:
    """Verify extracted facts against source text."""

    def __init__(self, provider: Any, model: str = "anthropic/claude-haiku-4-5-20251001"):
        self.provider = provider
        self.model = model

    async def check(self, fact: Fact, source_text: str) -> GroundingResult:
        # Skip grounding for user-stated facts
        if fact.extraction_type == ExtractionType.USER_STATED:
            return GroundingResult(grounded=True, confidence=1.0)

        try:
            response = await self.provider.chat(
                messages=[{
                    "role": "user",
                    "content": GROUNDING_PROMPT.format(
                        source_text=source_text[:2000],
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                    ),
                }],
                model=self.model,
                max_tokens=200,
                temperature=0.0,
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(text)
            return GroundingResult(
                grounded=parsed.get("grounded", False),
                evidence=parsed.get("evidence"),
                confidence=float(parsed.get("confidence_adjustment", 0.5)),
            )
        except Exception as e:
            logger.warning(f"Grounding check failed: {e}")
            # On failure, assume grounded with lower confidence
            return GroundingResult(grounded=True, confidence=0.5)


# ── Ingestion Gate 4: Coherence Check (DB + logic) ──────────────

EXCLUSIVE_PREDICATES = {
    "lives_in", "works_at", "current_role",
    "primary_language", "timezone", "uses_model",
    "current_project",
}


class CoherenceChecker:
    """Check if new fact contradicts existing facts."""

    def __init__(self, backend: Any):
        self.backend = backend

    async def check(self, fact: Fact) -> CoherenceResult:
        existing_facts = await self.backend.find_facts_by_subject_predicate(
            subject=fact.subject,
            predicate=fact.predicate,
            only_active=True,
        )

        if not existing_facts:
            return CoherenceResult(coherent=True)

        for existing in existing_facts:
            if existing["object"] != fact.object:
                if fact.predicate in EXCLUSIVE_PREDICATES:
                    return CoherenceResult(
                        coherent=False,
                        conflict_type="supersedes",
                        old_fact_id=existing["id"],
                        old_value=existing["object"],
                        new_value=fact.object,
                        action="invalidate_old",
                    )
                else:
                    return CoherenceResult(coherent=True, note="parallel_fact")

        return CoherenceResult(coherent=True)


# ── Recall Validation (at query time) ───────────────────────────

class RecallValidator:
    """Filter and rank facts before returning to agent."""

    async def validate_recall(
        self,
        query: str,
        results: list[SearchResult],
        mode: str = "normal",
    ) -> list[SearchResult]:
        validated = []

        for sr in results:
            fact = sr.fact
            issues = list(sr.issues)

            # 1. Skip expired
            if fact.valid_until is not None:
                continue

            # 2. Freshness
            if fact.last_confirmed:
                age_days = (datetime.now() - fact.last_confirmed).days
                if age_days > 180:
                    sr.score -= 0.2
                    issues.append(f"stale ({age_days}d)")

            # 3. Confidence
            if fact.confidence < 0.5:
                if mode == "strict":
                    continue
                sr.score -= 0.3
                issues.append(f"low confidence ({fact.confidence:.1f})")

            # 4. Source quality bonus
            if fact.extraction_type == ExtractionType.USER_STATED:
                sr.score += 0.3
            elif fact.extraction_type == ExtractionType.AUTO:
                sr.score += fact.confidence * 0.1

            sr.issues = issues
            validated.append(sr)

        # Sort by score descending
        validated.sort(key=lambda x: x.score, reverse=True)

        # Coherence: remove contradicting facts, keep freshest
        return self._check_recall_coherence(validated)

    def _check_recall_coherence(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove contradicting facts from recall results."""
        seen: dict[tuple[str, str], SearchResult] = {}
        final = []

        for sr in results:
            key = (sr.fact.subject, sr.fact.predicate)
            if key in seen:
                existing = seen[key]
                if existing.fact.object != sr.fact.object:
                    # Conflict — keep the one with higher score (already sorted)
                    sr.note = f"(superseded by: {existing.fact.object})"
                    continue
            seen[key] = sr
            final.append(sr)

        return final
