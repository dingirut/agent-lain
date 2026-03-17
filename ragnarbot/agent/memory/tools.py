"""Memory tools for the agent: search, store, forget, extract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ragnarbot.agent.tools.base import Tool

if TYPE_CHECKING:
    from ragnarbot.agent.memory.store import MemoryStore


class _MemoryToolMixin:
    """Common context tracking for memory tools."""

    _channel: str = ""
    _session_key: str = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._session_key = f"{channel}:{chat_id}"


class MemorySearchTool(_MemoryToolMixin, Tool):
    """Search semantic memory for facts."""

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search the agent's long-term semantic memory for facts about people, "
            "projects, preferences, events, and relationships. Returns scored results "
            "ranked by relevance. Use this to recall previously learned information."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        limit = kwargs.get("limit", 5)

        if not self._memory.has_vector:
            return "Semantic memory not available (Layer 1 not initialized)."

        results = await self._memory.search(query=query, limit=limit)
        if not results:
            return f"No facts found for: {query}"

        lines = [f"Found {len(results)} fact(s):"]
        for i, r in enumerate(results, 1):
            line = (
                f"{i}. [{r['category']}] {r['fact']} "
                f"(score: {r['score']}, confidence: {r['confidence']:.1f})"
            )
            if r.get("issues"):
                line += f" ⚠ {', '.join(r['issues'])}"
            lines.append(line)
        return "\n".join(lines)


class MemoryStoreTool(_MemoryToolMixin, Tool):
    """Explicitly store a fact in memory."""

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    @property
    def name(self) -> str:
        return "memory_store"

    @property
    def description(self) -> str:
        return (
            "Store a fact in long-term semantic memory. Use when the user explicitly "
            "asks you to remember something. The fact will be extracted, validated, "
            "and stored with high confidence."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or information to remember, in natural language",
                },
            },
            "required": ["content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        content = kwargs["content"]

        if not self._memory.has_vector:
            return "Semantic memory not available (Layer 1 not initialized)."

        result = await self._memory.store_user_fact(
            content=content,
            session_key=self._session_key,
            channel=self._channel,
        )
        if "error" in result:
            return f"Failed to store: {result['error']}"
        return f"Stored. {result.get('summary', '')}"


class MemoryForgetTool(_MemoryToolMixin, Tool):
    """Forget/invalidate facts from memory."""

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    @property
    def name(self) -> str:
        return "memory_forget"

    @property
    def description(self) -> str:
        return (
            "Forget (invalidate) facts from semantic memory. Use when the user "
            "asks you to forget something or when information is known to be outdated."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what to forget",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]

        if not self._memory.has_vector:
            return "Semantic memory not available (Layer 1 not initialized)."

        result = await self._memory.forget(query=query)
        if "error" in result:
            return f"Failed to forget: {result['error']}"
        return (
            f"Invalidated {result['invalidated']} fact(s) "
            f"(searched {result['searched']} candidates)."
        )


class MemoryExtractTool(_MemoryToolMixin, Tool):
    """Extract and store facts from a conversation excerpt."""

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    @property
    def name(self) -> str:
        return "memory_extract"

    @property
    def description(self) -> str:
        return (
            "Extract facts from a conversation excerpt and store them in memory. "
            "Use when you want to explicitly process a piece of text for fact extraction. "
            "Facts go through the full validation pipeline (grounding, deduplication, coherence)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Conversation text to extract facts from",
                },
                "skip_grounding": {
                    "type": "boolean",
                    "description": "Skip LLM grounding check (default false)",
                },
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        text = kwargs["text"]
        skip_grounding = kwargs.get("skip_grounding", False)

        if not self._memory.has_vector:
            return "Semantic memory not available (Layer 1 not initialized)."

        result = await self._memory.extract_and_store(
            text=text,
            session_key=self._session_key,
            channel=self._channel,
            skip_grounding=skip_grounding,
        )
        if "error" in result:
            return f"Extraction failed: {result['error']}"
        return f"Extraction complete. {result.get('summary', '')}"
