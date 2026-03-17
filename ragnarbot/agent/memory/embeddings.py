"""Embedding providers for semantic memory.

Fallback chain: Ollama (local) → Voyage API → LiteLLM → None (files-only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger


class EmbeddingProvider(ABC):
    """Base embedding provider interface."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Returns list of vectors."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Returns vector."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensionality."""
        ...


class OllamaEmbedding(EmbeddingProvider):
    """Ollama local embedding.

    Models: nomic-embed-text (768d), mxbai-embed-large (1024d), all-minilm (384d).
    """

    DIMS = {
        "nomic-embed-text": 768,
        "paraphrase-multilingual": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "paraphrase-multilingual",
    ):
        import httpx
        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120.0)
        self._dimensions = self.DIMS.get(model, 768)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            resp = await self._client.post("/api/embed", json={
                "model": self.model, "input": text,
            })
            resp.raise_for_status()
            results.append(resp.json()["embeddings"][0])
        return results

    async def embed_query(self, text: str) -> list[float]:
        resp = await self._client.post("/api/embed", json={
            "model": self.model, "input": text,
        })
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def close(self) -> None:
        await self._client.aclose()


class VoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding (recommended by Anthropic).

    Models: voyage-3 (1024d), voyage-3-large (1024d), voyage-code-3 (1024d).
    """

    def __init__(self, api_key: str, model: str = "voyage-3"):
        import httpx
        self.model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.voyageai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._dimensions = 1024

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post("/embeddings", json={
            "input": texts, "model": self.model, "input_type": "document",
        })
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    async def embed_query(self, text: str) -> list[float]:
        resp = await self._client.post("/embeddings", json={
            "input": [text], "model": self.model, "input_type": "query",
        })
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def close(self) -> None:
        await self._client.aclose()


class LiteLLMEmbedding(EmbeddingProvider):
    """Fallback: uses litellm for embedding through any configured provider."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._dimensions = 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import litellm
        response = await litellm.aembedding(model=self.model, input=texts)
        return [item["embedding"] for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        import litellm
        response = await litellm.aembedding(model=self.model, input=[text])
        return response.data[0]["embedding"]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def close(self) -> None:
        pass


def create_embedding_provider(
    provider: str | None = None,
    model: str | None = None,
    credentials: dict | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> EmbeddingProvider | None:
    """Factory for embedding provider.

    Priority: explicit provider → Ollama (local) → Voyage → LiteLLM → None.
    """
    creds = credentials or {}

    # Explicit provider
    if provider == "ollama":
        return OllamaEmbedding(base_url=ollama_base_url, model=model or "nomic-embed-text")
    if provider == "voyage":
        api_key = creds.get("voyage_api_key")
        if api_key:
            return VoyageEmbedding(api_key=api_key, model=model or "voyage-3")
        logger.warning("Voyage provider requested but no voyage_api_key in credentials")
        return None
    if provider == "litellm":
        return LiteLLMEmbedding(model=model or "text-embedding-3-small")

    # Auto-detect: Ollama first (local, free)
    if not provider:
        try:
            import httpx
            resp = httpx.get(f"{ollama_base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                target = model or "paraphrase-multilingual"
                # Check if model or its base name is available
                if any(target in m for m in models):
                    logger.info(f"Using Ollama embedding: {target}")
                    return OllamaEmbedding(base_url=ollama_base_url, model=target)
                logger.info(f"Ollama available but {target} not found (have: {models})")
        except Exception:
            pass

        # Voyage
        voyage_key = creds.get("voyage_api_key")
        if voyage_key:
            logger.info("Using Voyage embedding")
            return VoyageEmbedding(api_key=voyage_key, model=model or "voyage-3")

        # LiteLLM fallback (needs OPENAI_API_KEY or similar in env)
        openai_key = creds.get("openai_api_key")
        if openai_key:
            import os
            os.environ.setdefault("OPENAI_API_KEY", openai_key)
            logger.info("Using LiteLLM embedding (OpenAI)")
            return LiteLLMEmbedding(model=model or "text-embedding-3-small")

    logger.warning("No embedding provider available — memory will use files-only mode")
    return None
