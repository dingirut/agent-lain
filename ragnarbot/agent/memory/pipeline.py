"""Validated ingestion pipeline: extraction → validation → enrichment → storage."""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.embeddings import EmbeddingProvider
from ragnarbot.agent.memory.enrichment import MetadataEnricher
from ragnarbot.agent.memory.extraction import FactExtractor
from ragnarbot.agent.memory.models import (
    ExtractionType,
    Fact,
    FactCategory,
    IngestionReport,
    Source,
)
from ragnarbot.agent.memory.validation import (
    ALWAYS_IMPORTANT,
    CONDITIONAL,
    NOISE,
    NOISE_SUBJECTS,
    CoherenceChecker,
    DuplicateChecker,
    GroundingChecker,
    check_importance,
)


class IngestionPipeline:
    """Full pipeline: extract → validate → enrich → embed → store.

    Each fact passes through 4 validation gates before storage.
    """

    def __init__(
        self,
        extractor: FactExtractor,
        embedder: EmbeddingProvider,
        backend: Any,  # VectorBackend (pgvector)
        graph: Any | None,  # Neo4jBackend
        enricher: MetadataEnricher,
        grounding_checker: GroundingChecker,
        duplicate_checker: DuplicateChecker,
        coherence_checker: CoherenceChecker,
    ):
        self.extractor = extractor
        self.embedder = embedder
        self.backend = backend
        self.graph = graph
        self.enricher = enricher
        self.grounding = grounding_checker
        self.duplicates = duplicate_checker
        self.coherence = coherence_checker

    async def process(
        self,
        text: str,
        source: Source,
        extraction_type: ExtractionType = ExtractionType.AUTO,
        skip_grounding: bool = False,
    ) -> IngestionReport:
        """Run the full ingestion pipeline.

        Args:
            text: Source text (conversation excerpt).
            source: Source metadata for provenance.
            extraction_type: How the extraction was triggered.
            skip_grounding: Skip LLM grounding check (for realtime extraction).
        """
        report = IngestionReport()

        # 1. Extract facts
        facts = await self.extractor.extract_facts(text, source, extraction_type)
        report.extracted = len(facts)

        if not facts:
            return report

        # 2. Extract entities (for graph)
        entities = []
        if self.graph:
            entities = await self.extractor.extract_entities(facts)

        for fact in facts:
            try:
                await self._process_single_fact(
                    fact, text, source, report,
                    skip_grounding=skip_grounding,
                )
            except Exception as e:
                logger.error(f"Error processing fact '{fact.fact_text}': {e}")
                report.errors.append(str(e))

        # Store entities in graph
        if self.graph and entities:
            for ent in entities:
                try:
                    await self.graph.upsert_entity(
                        name=ent.get("name", ""),
                        entity_type=ent.get("entity_type", ""),
                        aliases=ent.get("aliases", []),
                    )
                except Exception as e:
                    logger.warning(f"Failed to store entity {ent}: {e}")

        logger.info(f"Ingestion complete: {report.summary()}")
        return report

    async def _process_single_fact(
        self,
        fact: Fact,
        source_text: str,
        source: Source,
        report: IngestionReport,
        skip_grounding: bool = False,
    ) -> None:
        """Process a single fact through all validation gates."""

        # Gate 1: Importance filter (free, no LLM)
        if not check_importance(fact):
            report.filtered_noise += 1
            return

        # Gate 2: Duplicate check (DB query)
        dup_result = await self.duplicates.check(fact)
        if dup_result.is_duplicate:
            if dup_result.action == "update_source":
                await self.backend.add_source_to_fact(
                    dup_result.existing_fact_id,
                    {"session": source.session_key, "timestamp": source.timestamp.isoformat()},
                )
            report.deduplicated += 1
            return

        # Gate 3: Grounding check (LLM call — expensive)
        if not skip_grounding:
            grounding = await self.grounding.check(fact, source_text)
            if not grounding.grounded:
                report.hallucinated += 1
                report.hallucination_log.append({
                    "fact": fact.fact_text,
                    "reason": "not grounded in source text",
                })
                return
            # Adjust confidence
            fact.confidence = min(fact.confidence, grounding.confidence)
            fact.evidence = grounding.evidence

        # Gate 4: Coherence check (DB + logic)
        coherence = await self.coherence.check(fact)
        if not coherence.coherent and coherence.action == "invalidate_old":
            await self.backend.invalidate_fact(coherence.old_fact_id)
            if self.graph:
                await self.graph.invalidate_relation(coherence.old_fact_id)
            report.superseded += 1

        # Enrich metadata (HQ, keywords, category)
        metadata = await self.enricher.enrich(fact)
        fact.hypothetical_questions = metadata.hypothetical_questions
        fact.keywords = metadata.keywords
        try:
            fact.category = FactCategory(metadata.category)
        except ValueError:
            pass

        # Generate embeddings (2x: fact + HQ)
        fact_embedding = await self.embedder.embed_query(fact.fact_text)
        hq_embedding = await self.embedder.embed_query(fact.hq_text) if fact.hypothetical_questions else None

        # Store in pgvector
        fact_id = str(uuid.uuid4())
        await self.backend.store_fact(
            fact_id=fact_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object_val=fact.object,
            fact_embedding=fact_embedding,
            hq_embedding=hq_embedding or fact_embedding,
            metadata={
                "confidence": str(fact.confidence),
                "original_confidence": str(fact.original_confidence),
                "extraction_type": fact.extraction_type.value,
                "evidence": fact.evidence or "",
                "source_session": fact.source_session or "",
                "category": fact.category.value,
                "hypothetical_questions": "|".join(fact.hypothetical_questions),
                "keywords": ",".join(fact.keywords),
                "time_context": fact.time_context or "",
                "sources": f'[{{"session": "{source.session_key}", "timestamp": "{source.timestamp.isoformat()}"}}]',
            },
        )

        # Store relation in graph
        if self.graph:
            await self.graph.upsert_relation(
                subject=fact.subject,
                predicate=fact.predicate,
                object_name=fact.object,
                fact_id=fact_id,
                confidence=fact.confidence,
            )

        report.stored += 1

    async def store_user_fact(
        self,
        content: str,
        source: Source,
    ) -> IngestionReport:
        """Shortcut for user-stated facts (skip grounding, high confidence)."""
        return await self.process(
            text=content,
            source=source,
            extraction_type=ExtractionType.USER_STATED,
            skip_grounding=True,
        )

    # ── Debug pipeline ──────────────────────────────────────────

    async def process_debug(
        self,
        text: str,
        overrides: dict[str, Any] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline with full trace for debugging.

        Args:
            text: Source text to process.
            overrides: Optional dict to override prompts/filter sets.
            dry_run: If True, don't persist to DB.

        Returns:
            Dict with per-stage trace data.
        """
        from ragnarbot.agent.memory.extraction import EXTRACTION_PROMPT
        from ragnarbot.agent.memory.enrichment import HQ_PROMPT
        from ragnarbot.agent.memory.validation import GROUNDING_PROMPT

        ov = overrides or {}
        source = Source(session_key="debug", channel="debug")
        trace: dict[str, Any] = {"input_text": text, "stages": {}}

        # Build override sets
        ov_noise = set(ov["noise_predicates"]) if "noise_predicates" in ov else None
        ov_noise_subj = set(ov["noise_subjects"]) if "noise_subjects" in ov else None
        ov_always = set(ov["always_important"]) if "always_important" in ov else None

        # ── Stage 1: Extraction ──
        extraction_prompt = ov.get("extraction_prompt", EXTRACTION_PROMPT)
        try:
            response = await self.extractor.provider.chat(
                messages=[{
                    "role": "user",
                    "content": extraction_prompt.format(text=text[:4000]),
                }],
                model=self.extractor.model,
                max_tokens=4000,
                temperature=0.0,
            )
            raw_response = response.content or "[]"
        except Exception as e:
            raw_response = f"ERROR: {e}"

        from ragnarbot.agent.memory.extraction import _parse_json_response
        raw_facts = _parse_json_response(raw_response)
        facts: list[Fact] = []
        for raw in raw_facts:
            if not all(k in raw for k in ("subject", "predicate", "object")):
                continue
            fact = Fact(
                subject=raw["subject"].strip(),
                predicate=raw["predicate"].strip(),
                object=raw["object"].strip(),
                confidence=float(raw.get("confidence", 0.7)),
                original_confidence=float(raw.get("confidence", 0.7)),
                extraction_type=ExtractionType.AUTO,
                category=raw.get("category", "other"),
                time_context=raw.get("time_context"),
            )
            try:
                fact.category = FactCategory(fact.category)
            except ValueError:
                fact.category = FactCategory.OTHER
            facts.append(fact)

        trace["stages"]["extraction"] = {
            "prompt_used": extraction_prompt,
            "raw_response": raw_response,
            "facts": [
                {"subject": f.subject, "predicate": f.predicate, "object": f.object,
                 "confidence": f.confidence, "category": f.category.value}
                for f in facts
            ],
        }

        if not facts:
            trace["stages"]["gate1_importance"] = {"results": []}
            trace["stages"]["gate2_duplicates"] = {"results": []}
            trace["stages"]["gate3_grounding"] = {"results": []}
            trace["stages"]["gate4_coherence"] = {"results": []}
            trace["stages"]["enrichment"] = {"results": []}
            trace["stages"]["final"] = {"stored": 0, "rejected": 0, "facts": []}
            return trace

        # ── Stage 2: Gate 1 — Importance ──
        gate1_results = []
        surviving_facts: list[Fact] = []
        for fact in facts:
            passed, reason = _check_importance_debug(
                fact,
                noise=ov_noise,
                noise_subjects=ov_noise_subj,
                always_important=ov_always,
            )
            gate1_results.append({
                "fact": fact.fact_text,
                "passed": passed,
                "reason": reason,
            })
            if passed:
                surviving_facts.append(fact)
        trace["stages"]["gate1_importance"] = {"results": gate1_results}

        # ── Stage 3: Gate 2 — Duplicates ──
        gate2_results = []
        after_dedup: list[Fact] = []
        for fact in surviving_facts:
            try:
                dup = await self.duplicates.check(fact)
                gate2_results.append({
                    "fact": fact.fact_text,
                    "is_duplicate": dup.is_duplicate,
                    "action": dup.action,
                    "similarity": round(dup.similarity, 3),
                })
                if not dup.is_duplicate:
                    after_dedup.append(fact)
            except Exception as e:
                gate2_results.append({
                    "fact": fact.fact_text,
                    "is_duplicate": False,
                    "error": str(e),
                })
                after_dedup.append(fact)
        trace["stages"]["gate2_duplicates"] = {"results": gate2_results}

        # ── Stage 4: Gate 3 — Grounding ──
        grounding_enabled = ov.get("grounding_enabled", True)
        gate3_results = []
        after_grounding: list[Fact] = []
        for fact in after_dedup:
            if not grounding_enabled:
                gate3_results.append({
                    "fact": fact.fact_text,
                    "grounded": True,
                    "skipped": True,
                })
                after_grounding.append(fact)
                continue
            try:
                gr = await self.grounding.check(fact, text)
                gate3_results.append({
                    "fact": fact.fact_text,
                    "grounded": gr.grounded,
                    "evidence": gr.evidence,
                    "confidence": round(gr.confidence, 3),
                })
                if gr.grounded:
                    fact.confidence = min(fact.confidence, gr.confidence)
                    fact.evidence = gr.evidence
                    after_grounding.append(fact)
            except Exception as e:
                gate3_results.append({
                    "fact": fact.fact_text,
                    "grounded": True,
                    "error": str(e),
                })
                after_grounding.append(fact)
        trace["stages"]["gate3_grounding"] = {"results": gate3_results}

        # ── Stage 5: Gate 4 — Coherence ──
        gate4_results = []
        after_coherence: list[Fact] = []
        for fact in after_grounding:
            try:
                coh = await self.coherence.check(fact)
                result_entry: dict[str, Any] = {
                    "fact": fact.fact_text,
                    "coherent": coh.coherent,
                }
                if not coh.coherent:
                    result_entry["conflict_type"] = coh.conflict_type
                    result_entry["old_value"] = coh.old_value
                    result_entry["new_value"] = coh.new_value
                    result_entry["action"] = coh.action
                gate4_results.append(result_entry)
                after_coherence.append(fact)
            except Exception as e:
                gate4_results.append({
                    "fact": fact.fact_text,
                    "coherent": True,
                    "error": str(e),
                })
                after_coherence.append(fact)
        trace["stages"]["gate4_coherence"] = {"results": gate4_results}

        # ── Stage 6: Enrichment ──
        enrichment_results = []
        for fact in after_coherence:
            try:
                meta = await self.enricher.enrich(fact)
                fact.hypothetical_questions = meta.hypothetical_questions
                fact.keywords = meta.keywords
                try:
                    fact.category = FactCategory(meta.category)
                except ValueError:
                    pass
                enrichment_results.append({
                    "fact": fact.fact_text,
                    "hq": meta.hypothetical_questions,
                    "keywords": meta.keywords,
                    "category": meta.category,
                })
            except Exception as e:
                enrichment_results.append({
                    "fact": fact.fact_text,
                    "error": str(e),
                })
        trace["stages"]["enrichment"] = {"results": enrichment_results}

        # ── Stage 7: Store (if not dry_run) ──
        stored_count = 0
        final_facts = []
        for fact in after_coherence:
            entry = {
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": round(fact.confidence, 3),
                "category": fact.category.value,
                "stored": False,
            }
            if not dry_run:
                try:
                    fact_embedding = await self.embedder.embed_query(fact.fact_text)
                    hq_embedding = (
                        await self.embedder.embed_query(fact.hq_text)
                        if fact.hypothetical_questions else None
                    )
                    fact_id = str(uuid.uuid4())
                    await self.backend.store_fact(
                        fact_id=fact_id,
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object_val=fact.object,
                        fact_embedding=fact_embedding,
                        hq_embedding=hq_embedding or fact_embedding,
                        metadata={
                            "confidence": str(fact.confidence),
                            "original_confidence": str(fact.original_confidence),
                            "extraction_type": fact.extraction_type.value,
                            "evidence": fact.evidence or "",
                            "source_session": source.session_key,
                            "category": fact.category.value,
                            "hypothetical_questions": "|".join(fact.hypothetical_questions),
                            "keywords": ",".join(fact.keywords),
                            "time_context": fact.time_context or "",
                            "sources": f'[{{"session": "debug", "timestamp": "{source.timestamp.isoformat()}"}}]',
                        },
                    )
                    if self.graph:
                        await self.graph.upsert_relation(
                            subject=fact.subject,
                            predicate=fact.predicate,
                            object_name=fact.object,
                            fact_id=fact_id,
                            confidence=fact.confidence,
                        )
                    entry["stored"] = True
                    stored_count += 1
                except Exception as e:
                    entry["error"] = str(e)
            final_facts.append(entry)

        # ── Entity extraction for graph (when storing) ──
        entities_stored = []
        if not dry_run and self.graph and after_coherence:
            try:
                entities = await self.extractor.extract_entities(after_coherence)
                for ent in entities:
                    try:
                        await self.graph.upsert_entity(
                            name=ent.get("name", ""),
                            entity_type=ent.get("entity_type", ""),
                            aliases=ent.get("aliases", []),
                        )
                        entities_stored.append(ent)
                    except Exception as e:
                        logger.warning(f"Failed to store entity {ent}: {e}")
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        rejected = len(facts) - len(after_coherence)
        trace["stages"]["final"] = {
            "stored": stored_count,
            "passed_all_gates": len(after_coherence),
            "rejected": rejected,
            "facts": final_facts,
            "entities": entities_stored,
        }
        return trace


def _check_importance_debug(
    fact: Fact,
    noise: set[str] | None = None,
    noise_subjects: set[str] | None = None,
    always_important: set[str] | None = None,
) -> tuple[bool, str]:
    """Check importance with override support. Returns (passed, reason)."""
    _noise = noise if noise is not None else NOISE
    _noise_subj = noise_subjects if noise_subjects is not None else NOISE_SUBJECTS
    _always = always_important if always_important is not None else ALWAYS_IMPORTANT

    predicate = fact.predicate.lower().strip()
    subject_lower = fact.subject.lower().strip()

    if subject_lower in _noise_subj:
        return False, f"subject '{subject_lower}' in NOISE_SUBJECTS"
    if predicate in _noise:
        return False, f"predicate '{predicate}' in NOISE"
    if predicate in _always:
        return True, f"predicate '{predicate}' in ALWAYS_IMPORTANT"
    if predicate in CONDITIONAL:
        if fact.confidence > 0.8:
            return True, f"predicate '{predicate}' in CONDITIONAL, confidence {fact.confidence:.2f} > 0.8"
        return False, f"predicate '{predicate}' in CONDITIONAL, confidence {fact.confidence:.2f} <= 0.8"
    if fact.confidence < 0.75:
        return False, f"confidence {fact.confidence:.2f} < 0.75"
    if len(fact.subject.strip()) < 2 or len(fact.object.strip()) < 2:
        return False, "subject or object too short (< 2 chars)"
    return True, "passed (unknown predicate, confidence >= 0.75)"
