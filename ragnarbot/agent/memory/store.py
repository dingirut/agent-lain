"""Updated MemoryStore: Layer 0 (files) + Layer 1 (vector/graph)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.embeddings import EmbeddingProvider, create_embedding_provider
from ragnarbot.agent.memory.models import ExtractionType, Source
from ragnarbot.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """Memory system combining file-based memory with semantic vector memory.

    Layer 0: File-based (MEMORY.md + daily notes) — always available.
    Layer 1: Vector memory (pgvector + Neo4j) — optional, enabled when
             embedding provider and database are available.
    """

    def __init__(
        self,
        workspace: Path,
        provider: Any | None = None,  # LLMProvider for extraction/validation
        embedding_config: dict | None = None,
        database_url: str | None = None,
        neo4j_url: str | None = None,
        neo4j_auth: tuple[str, str] | None = None,
        extraction_model: str = "anthropic/claude-haiku-4-5-20251001",
        validation_model: str = "anthropic/claude-haiku-4-5-20251001",
        enrichment_model: str = "anthropic/claude-haiku-4-5-20251001",
        auto_extract: bool = True,
        auto_inject: bool = True,
        similarity_threshold: float = 0.1,
    ):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"

        # Layer 1 config
        self._provider = provider
        self._embedding_config = embedding_config or {}
        self._database_url = database_url or "postgresql://ragnarbot:ragnarbot@localhost:5432/ragnarbot"
        self._neo4j_url = neo4j_url or "bolt://localhost:7687"
        self._neo4j_auth = neo4j_auth or ("neo4j", "ragnarbot")

        # Hot-reloadable settings
        self.extraction_model = extraction_model
        self.validation_model = validation_model
        self.enrichment_model = enrichment_model
        self.auto_extract = auto_extract
        self.auto_inject = auto_inject
        self.similarity_threshold = similarity_threshold

        # Layer 1 components (lazy init)
        self._embedder: EmbeddingProvider | None = None
        self._backend: Any = None  # PgvectorBackend
        self._graph: Any = None  # Neo4jBackend
        self._pipeline: Any = None  # IngestionPipeline
        self._retriever: Any = None  # HybridRetriever
        self._initialized = False
        self._init_failed = False

    @property
    def has_vector(self) -> bool:
        """Whether vector memory (Layer 1) is available."""
        return self._initialized and self._embedder is not None

    async def initialize(self) -> bool:
        """Initialize Layer 1 components. Returns True if successful."""
        if self._initialized:
            return True
        if self._init_failed:
            return False

        try:
            # 1. Embedding provider
            self._embedder = create_embedding_provider(
                provider=self._embedding_config.get("provider"),
                model=self._embedding_config.get("model"),
                credentials=self._embedding_config.get("credentials", {}),
                ollama_base_url=self._embedding_config.get("ollama_base_url", "http://localhost:11434"),
            )
            if not self._embedder:
                logger.warning("No embedding provider — Layer 1 disabled")
                self._init_failed = True
                return False

            # 2. pgvector backend
            from ragnarbot.agent.memory.backends.pgvector_backend import PgvectorBackend
            self._backend = PgvectorBackend(
                database_url=self._database_url,
                dimensions=self._embedder.dimensions,
            )
            await self._backend.initialize()

            # 3. Neo4j graph (optional)
            try:
                from ragnarbot.agent.memory.graph.neo4j_backend import Neo4jBackend
                self._graph = Neo4jBackend(
                    uri=self._neo4j_url,
                    auth=self._neo4j_auth,
                )
                await self._graph.initialize()
                logger.info("Neo4j graph backend initialized")
            except Exception as e:
                logger.warning(f"Neo4j unavailable, graph features disabled: {e}")
                self._graph = None

            # 4. Build pipeline & retriever
            self._build_pipeline()
            self._build_retriever()

            self._initialized = True
            logger.info(
                f"Memory Layer 1 initialized: embedder={type(self._embedder).__name__}, "
                f"dims={self._embedder.dimensions}, graph={'yes' if self._graph else 'no'}"
            )
            return True

        except Exception as e:
            logger.error(f"Memory Layer 1 initialization failed: {e}")
            self._init_failed = False  # allow retry
            return False

    def _build_pipeline(self) -> None:
        """Build or rebuild the ingestion pipeline with current model settings."""
        from ragnarbot.agent.memory.enrichment import MetadataEnricher
        from ragnarbot.agent.memory.extraction import FactExtractor
        from ragnarbot.agent.memory.pipeline import IngestionPipeline
        from ragnarbot.agent.memory.validation import (
            CoherenceChecker,
            DuplicateChecker,
            GroundingChecker,
        )

        self._pipeline = IngestionPipeline(
            extractor=FactExtractor(provider=self._provider, model=self.extraction_model),
            embedder=self._embedder,
            backend=self._backend,
            graph=self._graph,
            enricher=MetadataEnricher(provider=self._provider, model=self.enrichment_model),
            grounding_checker=GroundingChecker(provider=self._provider, model=self.validation_model),
            duplicate_checker=DuplicateChecker(backend=self._backend, embedder=self._embedder),
            coherence_checker=CoherenceChecker(backend=self._backend),
        )

    def _build_retriever(self) -> None:
        """Build or rebuild the hybrid retriever."""
        from ragnarbot.agent.memory.retriever import HybridRetriever
        from ragnarbot.agent.memory.validation import RecallValidator

        self._retriever = HybridRetriever(
            embedder=self._embedder,
            backend=self._backend,
            graph=self._graph,
            recall_validator=RecallValidator(),
        )

    def reload_models(
        self,
        extraction_model: str | None = None,
        validation_model: str | None = None,
        enrichment_model: str | None = None,
    ) -> None:
        """Hot-reload LLM models used for extraction/validation/enrichment."""
        changed = False
        if extraction_model and extraction_model != self.extraction_model:
            self.extraction_model = extraction_model
            changed = True
        if validation_model and validation_model != self.validation_model:
            self.validation_model = validation_model
            changed = True
        if enrichment_model and enrichment_model != self.enrichment_model:
            self.enrichment_model = enrichment_model
            changed = True

        if changed and self._initialized:
            self._build_pipeline()
            logger.info(
                f"Memory models reloaded: extract={self.extraction_model}, "
                f"validate={self.validation_model}, enrich={self.enrichment_model}"
            )

    # ── Layer 0: File-based memory ──────────────────────────────────

    def get_day_file(self, date_str: str) -> Path:
        """Get path to a daily memory file."""
        return self.memory_dir / f"{date_str}.md"

    def get_today_file(self) -> Path:
        return self.get_day_file(today_date())

    def get_yesterday_file(self) -> Path:
        from datetime import datetime, timedelta
        yesterday = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        return self.get_day_file(yesterday)

    def read_day(self, date_str: str) -> str:
        """Read a daily memory file."""
        day_file = self.get_day_file(date_str)
        if day_file.exists():
            return day_file.read_text(encoding="utf-8")
        return ""

    def write_day(self, date_str: str, content: str) -> None:
        """Write the full contents of a daily memory file."""
        self.get_day_file(date_str).write_text(content, encoding="utf-8")

    def read_today(self) -> str:
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""

    def append_today(self, content: str) -> None:
        today_file = self.get_today_file()
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            from datetime import datetime
            header = f"# {today_date()}\n\n"
            content = header + content
        today_file.write_text(content, encoding="utf-8")

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def get_recent_memories(self, days: int = 7) -> str:
        from datetime import datetime, timedelta
        memories = []
        today = datetime.now().date()
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)
        return "\n\n---\n\n".join(memories)

    def list_memory_files(self) -> list[Path]:
        if not self.memory_dir.exists():
            return []
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    def get_memory_context(self) -> str:
        """Get file-based memory context (Layer 0)."""
        parts = []
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)
        yesterday = self.read_day(self.get_yesterday_file().stem)
        if yesterday:
            parts.append("## Yesterday's Notes\n" + yesterday)
        return "\n\n".join(parts) if parts else ""

    # ── Layer 1: Vector memory ──────────────────────────────────────

    async def recall_for_context(
        self,
        query: str,
        max_results: int = 3,
        max_chars: int = 2000,
    ) -> str:
        """Retrieve relevant facts for system prompt injection.

        Returns formatted string or empty if Layer 1 is not available.
        """
        if not self.has_vector or not self._retriever:
            return ""
        try:
            return await self._retriever.search_for_context(
                query=query, max_results=max_results, max_chars=max_chars,
            )
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return ""

    async def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "normal",
    ) -> list[dict]:
        """Search vector memory. Returns list of formatted results."""
        if not self.has_vector or not self._retriever:
            return []
        try:
            results = await self._retriever.search(
                query=query, limit=limit,
                score_threshold=self.similarity_threshold, mode=mode,
            )
            return [
                {
                    "fact": r.fact.fact_text,
                    "subject": r.fact.subject,
                    "predicate": r.fact.predicate,
                    "object": r.fact.object,
                    "score": round(r.score, 3),
                    "category": r.fact.category.value,
                    "confidence": r.fact.confidence,
                    "sources": r.sources,
                    "issues": r.issues,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    async def extract_and_store(
        self,
        text: str,
        session_key: str = "",
        channel: str = "",
        extraction_type: ExtractionType = ExtractionType.AUTO,
        skip_grounding: bool = False,
    ) -> dict:
        """Extract facts from text and store them.

        Returns ingestion report as dict.
        """
        if not self.has_vector or not self._pipeline:
            return {"error": "Vector memory not available"}
        try:
            from datetime import datetime
            source = Source(
                session_key=session_key,
                channel=channel,
                timestamp=datetime.now(),
            )
            report = await self._pipeline.process(
                text=text,
                source=source,
                extraction_type=extraction_type,
                skip_grounding=skip_grounding,
            )
            return {
                "extracted": report.extracted,
                "stored": report.stored,
                "filtered_noise": report.filtered_noise,
                "deduplicated": report.deduplicated,
                "hallucinated": report.hallucinated,
                "superseded": report.superseded,
                "summary": report.summary(),
            }
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return {"error": str(e)}

    async def store_user_fact(
        self,
        content: str,
        session_key: str = "",
        channel: str = "",
    ) -> dict:
        """Store a user-stated fact (skip grounding, high confidence)."""
        return await self.extract_and_store(
            text=content,
            session_key=session_key,
            channel=channel,
            extraction_type=ExtractionType.USER_STATED,
            skip_grounding=True,
        )

    async def forget(self, query: str) -> dict:
        """Find and invalidate facts matching query."""
        if not self.has_vector or not self._backend:
            return {"error": "Vector memory not available"}
        try:
            results = await self._retriever.search(query=query, limit=5)
            invalidated = 0
            for r in results:
                if r.score > 0.3 and r.fact.id:
                    await self._backend.invalidate_fact(r.fact.id)
                    if self._graph:
                        await self._graph.invalidate_relation(r.fact.id)
                    invalidated += 1
            return {"invalidated": invalidated, "searched": len(results)}
        except Exception as e:
            logger.error(f"Memory forget failed: {e}")
            return {"error": str(e)}

    def trigger_background_extraction(
        self,
        text: str,
        session_key: str = "",
        channel: str = "",
    ) -> None:
        """Fire-and-forget extraction in background (non-blocking)."""
        if not self.has_vector or not self.auto_extract:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._background_extract(text, session_key, channel))
        except RuntimeError:
            pass  # no event loop

    async def _background_extract(
        self,
        text: str,
        session_key: str,
        channel: str,
    ) -> None:
        """Background extraction task."""
        try:
            result = await self.extract_and_store(
                text=text,
                session_key=session_key,
                channel=channel,
                extraction_type=ExtractionType.AUTO,
                skip_grounding=False,
            )
            if result.get("stored", 0) > 0:
                logger.info(f"Background extraction: {result.get('summary', '')}")
        except Exception as e:
            logger.debug(f"Background extraction error: {e}")

    async def close(self) -> None:
        """Cleanup Layer 1 resources."""
        if self._backend and hasattr(self._backend, "close"):
            await self._backend.close()
        if self._graph and hasattr(self._graph, "close"):
            await self._graph.close()
        if self._embedder and hasattr(self._embedder, "close"):
            await self._embedder.close()
