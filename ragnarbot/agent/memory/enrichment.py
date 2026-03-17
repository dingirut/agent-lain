"""Metadata enrichment: hypothetical questions, keywords, category."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.models import EnrichedMetadata, Fact

HQ_PROMPT = """For the following fact, generate 2-4 natural questions
that a user might ask, and this fact would be the answer.

Fact: {subject} {predicate} {object}

Return JSON: {{"questions": ["...", "..."], "keywords": ["...", "..."], "category": "..."}}

Categories: personal, project, technical, preference, relationship, event, other

Example:
Fact: Enoch works_on ragnarbot
Questions: ["What project does Enoch work on?", "Who develops ragnarbot?",
            "What is Enoch building?"]
Keywords: ["enoch", "ragnarbot", "project", "developer"]
Category: "project"
"""


class MetadataEnricher:
    """Enrich facts with hypothetical questions and keywords for better retrieval."""

    def __init__(self, provider: Any, model: str = "anthropic/claude-haiku-4-5-20251001"):
        self.provider = provider
        self.model = model

    async def enrich(self, fact: Fact) -> EnrichedMetadata:
        """Generate hypothetical questions, keywords, and category for a fact."""
        try:
            response = await self.provider.chat(
                messages=[{
                    "role": "user",
                    "content": HQ_PROMPT.format(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                    ),
                }],
                model=self.model,
                max_tokens=300,
                temperature=0.0,
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(text)
            return EnrichedMetadata(
                hypothetical_questions=parsed.get("questions", []),
                keywords=parsed.get("keywords", []),
                category=parsed.get("category", fact.category.value if hasattr(fact.category, "value") else str(fact.category)),
            )
        except Exception as e:
            logger.warning(f"Metadata enrichment failed: {e}")
            # Fallback: generate basic metadata without LLM
            return EnrichedMetadata(
                hypothetical_questions=[
                    f"What does {fact.subject} {fact.predicate.replace('_', ' ')}?",
                ],
                keywords=[
                    w.lower() for w in (fact.subject.split() + fact.object.split())
                    if len(w) > 2
                ],
                category=fact.category.value if hasattr(fact.category, "value") else str(fact.category),
            )

    async def enrich_batch(self, facts: list[Fact]) -> list[EnrichedMetadata]:
        """Enrich multiple facts. Processes sequentially to avoid rate limits."""
        results = []
        for fact in facts:
            meta = await self.enrich(fact)
            results.append(meta)
        return results
