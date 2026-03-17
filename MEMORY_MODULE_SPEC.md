# Специфікація модуля семантичної пам'яті для Ragnarbot

> Документ для кодер-агента. Містить повну специфікацію для інтеграції семантичної пам'яті
> в ragnarbot на основі підходу PentAGI (pgvector + embeddings), адаптованого під
> архітектуру ragnarbot.

---

## 1. Загальний огляд

### 1.1. Що є зараз

Поточна пам'ять ragnarbot — це файлова система:

```
~/.ragnarbot/workspace/
├── memory/
│   ├── MEMORY.md           ← довгострокова (один файл, plain Markdown)
│   ├── 2026-03-10.md       ← щоденна нотатка
│   └── 2026-03-11.md
```

**Клас:** `ragnarbot/agent/memory.py` → `MemoryStore`

**Проблеми:**
- Вся пам'ять вставляється в system prompt цілком (`get_memory_context()`)
- Немає семантичного пошуку — тільки повне прочитання файлів
- Не масштабується: 100 KB пам'яті = +100 KB до кожного запиту
- Немає метаданих, типізації, ранжування релевантності
- Агент не має інструментів для роботи з пам'яттю

### 1.2. Що потрібно

Тришарова пам'ять з graceful degradation:

| Шар | Технологія | Коли доступний |
|-----|-----------|----------------|
| **Layer 0** | Файли (Markdown) | Завжди — zero dependencies |
| **Layer 1** | pgvector (PostgreSQL) | Якщо `memory.backend` = `pgvector` |
| **Layer 2** | ChromaDB (вбудований) | Якщо `memory.backend` = `chroma` (альтернатива без PostgreSQL) |

**Принцип:** Якщо векторна БД недоступна — система працює як зараз (файли). Якщо доступна — додається семантичний пошук. Агент отримує інструменти для роботи з пам'яттю.

---

## 2. Конфігурація

### 2.1. Додати в `config/schema.py`

```python
class MemoryBackend(str, Enum):
    FILES = "files"          # Поточна поведінка (default)
    PGVECTOR = "pgvector"    # PostgreSQL + pgvector
    CHROMA = "chroma"        # ChromaDB (вбудований, без зовнішніх залежностей)


class EmbeddingProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    # Провайдер визначається автоматично з наявних credentials, якщо не вказано


class MemoryConfig(BaseModel):
    backend: MemoryBackend = MemoryBackend.FILES

    # pgvector settings (тільки якщо backend = pgvector)
    database_url: Optional[str] = None  # postgresql://user:pass@host:5432/ragnarbot

    # chroma settings (тільки якщо backend = chroma)
    chroma_path: Optional[str] = None   # default: ~/.ragnarbot/chroma/

    # Embedding settings
    embedding_provider: Optional[EmbeddingProvider] = None  # auto-detect якщо None
    embedding_model: Optional[str] = None                   # default за провайдером

    # Search settings
    similarity_threshold: float = 0.2    # мінімальна cosine similarity
    max_results: int = 5                 # максимум результатів на запит

    # Chunking settings
    chunk_size: int = 2000               # розмір чанка в символах
    chunk_overlap: int = 100             # перекриття між чанками

    # Context injection (скільки пам'яті додавати в system prompt)
    auto_inject: bool = True             # автоматично додавати релевантну пам'ять
    auto_inject_max_results: int = 3     # скільки фактів автоматично додавати
    auto_inject_max_tokens: int = 2000   # ліміт токенів для auto-inject
```

Додати в головний `Config`:
```python
class Config(BaseModel):
    # ... existing fields ...
    memory: MemoryConfig = MemoryConfig()
```

### 2.2. Конфіг JSON приклад

```json
{
  "memory": {
    "backend": "chroma",
    "embeddingProvider": "openai",
    "embeddingModel": "text-embedding-3-small",
    "similarityThreshold": 0.2,
    "maxResults": 5,
    "chunkSize": 2000,
    "chunkOverlap": 100,
    "autoInject": true,
    "autoInjectMaxResults": 3,
    "autoInjectMaxTokens": 2000
  }
}
```

---

## 3. Архітектура модуля

### 3.1. Структура файлів

```
ragnarbot/agent/memory/
├── __init__.py              # re-export MemoryStore
├── store.py                 # MemoryStore — головний клас (оновлений)
├── vector_store.py          # VectorMemoryStore — абстракція над vector backends
├── backends/
│   ├── __init__.py
│   ├── base.py              # VectorBackend ABC
│   ├── pgvector_backend.py  # PostgreSQL + pgvector
│   └── chroma_backend.py    # ChromaDB (вбудований)
├── embeddings.py            # EmbeddingProvider — обгортка над LLM embeddings
├── chunker.py               # DocumentChunker — розбиття тексту на чанки
├── models.py                # MemoryDocument, SearchResult, DocType dataclasses
└── tools.py                 # LLM tools: memory_search, memory_store, memory_forget
```

### 3.2. Діаграма потоку

```
Agent Loop
  │
  ├─ build_system_prompt()
  │   └─ memory.get_context_for_query(user_message)  ← НОВЕ: semantic search
  │       ├─ [vector backend available] → similarity_search(query, top_k=3)
  │       └─ [files only] → get_memory_context() (як зараз)
  │
  ├─ LLM call з tools: [..., memory_search, memory_store, memory_forget]
  │
  └─ Tool execution:
      ├─ memory_search(query, doc_type, limit) → semantic search
      ├─ memory_store(content, doc_type, tags, question) → embed + store
      └─ memory_forget(query, doc_type) → видалення за запитом
```

---

## 4. Моделі даних

### 4.1. `models.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    """Типи документів у пам'яті.

    Адаптовано з PentAGI (memory, guide, answer, code) під загальне використання.
    """
    MEMORY = "memory"          # Загальні факти, спостереження, результати
    GUIDE = "guide"            # Покрокові інструкції, how-to
    ANSWER = "answer"          # Відповіді на запити, результати досліджень
    CODE = "code"              # Кодові зразки, скрипти, конфігурації
    CONVERSATION = "conversation"  # Стислі фрагменти важливих розмов


class GuideType(str, Enum):
    INSTALL = "install"
    CONFIGURE = "configure"
    USE = "use"
    DEVELOP = "develop"
    TROUBLESHOOT = "troubleshoot"
    OTHER = "other"


class AnswerType(str, Enum):
    GUIDE = "guide"
    RESEARCH = "research"
    CODE = "code"
    TOOL = "tool"
    OTHER = "other"


@dataclass
class MemoryDocument:
    """Один документ у пам'яті (може бути чанком більшого тексту)."""

    content: str                          # Текстовий вміст
    doc_type: DocType = DocType.MEMORY    # Тип документа

    # Metadata
    question: Optional[str] = None        # Питання, на яке відповідає документ
    tags: list[str] = field(default_factory=list)
    source_tool: Optional[str] = None     # Інструмент, що створив (exec, web_search...)
    source_description: Optional[str] = None

    # Типо-специфічні
    guide_type: Optional[str] = None      # Для DocType.GUIDE
    answer_type: Optional[str] = None     # Для DocType.ANSWER
    code_lang: Optional[str] = None       # Для DocType.CODE
    code_description: Optional[str] = None
    code_explanation: Optional[str] = None

    # Session context
    session_key: Optional[str] = None     # Ключ сеансу (telegram:123, web:456)
    channel: Optional[str] = None         # Канал (telegram, web)

    # Auto-populated
    created_at: datetime = field(default_factory=datetime.now)
    chunk_index: int = 0                  # Індекс чанка (якщо розбито)
    chunk_total: int = 1                  # Загальна кількість чанків
    total_size: int = 0                   # Загальний розмір оригіналу в байтах

    def to_metadata(self) -> dict[str, str]:
        """Конвертує в metadata dict для vector store.

        ВАЖЛИВО: Всі значення МАЮТЬ бути рядками — вимога pgvector.
        """
        meta = {
            "doc_type": self.doc_type.value,
            "created_at": self.created_at.isoformat(),
            "chunk_index": str(self.chunk_index),
            "chunk_total": str(self.chunk_total),
            "total_size": str(self.total_size),
        }
        if self.question:
            meta["question"] = self.question
        if self.tags:
            meta["tags"] = ",".join(self.tags)
        if self.source_tool:
            meta["source_tool"] = self.source_tool
        if self.source_description:
            meta["source_description"] = self.source_description
        if self.guide_type:
            meta["guide_type"] = self.guide_type
        if self.answer_type:
            meta["answer_type"] = self.answer_type
        if self.code_lang:
            meta["code_lang"] = self.code_lang
        if self.code_description:
            meta["code_description"] = self.code_description
        if self.session_key:
            meta["session_key"] = self.session_key
        if self.channel:
            meta["channel"] = self.channel
        return meta

    @classmethod
    def from_metadata(cls, content: str, metadata: dict[str, str]) -> MemoryDocument:
        """Відновлює з metadata dict (з vector store)."""
        return cls(
            content=content,
            doc_type=DocType(metadata.get("doc_type", "memory")),
            question=metadata.get("question"),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            source_tool=metadata.get("source_tool"),
            source_description=metadata.get("source_description"),
            guide_type=metadata.get("guide_type"),
            answer_type=metadata.get("answer_type"),
            code_lang=metadata.get("code_lang"),
            code_description=metadata.get("code_description"),
            session_key=metadata.get("session_key"),
            channel=metadata.get("channel"),
            created_at=datetime.fromisoformat(metadata["created_at"]) if "created_at" in metadata else datetime.now(),
            chunk_index=int(metadata.get("chunk_index", 0)),
            chunk_total=int(metadata.get("chunk_total", 1)),
            total_size=int(metadata.get("total_size", 0)),
        )


@dataclass
class SearchResult:
    """Результат семантичного пошуку."""
    document: MemoryDocument
    score: float                  # Cosine similarity (0.0 — 1.0)

    def format_for_llm(self, index: int) -> str:
        """Формат результату для LLM контексту.

        Адаптовано з PentAGI memory.go format.
        """
        doc = self.document
        parts = [f"# Memory Fact {index + 1} (relevance: {self.score:.0%})"]

        if doc.doc_type != DocType.MEMORY:
            parts.append(f"**Type:** {doc.doc_type.value}")
        if doc.question:
            parts.append(f"**Question:** {doc.question}")
        if doc.source_tool:
            parts.append(f"**Source tool:** {doc.source_tool}")
        if doc.tags:
            parts.append(f"**Tags:** {', '.join(doc.tags)}")
        if doc.code_lang:
            parts.append(f"**Language:** {doc.code_lang}")
        if doc.guide_type:
            parts.append(f"**Guide type:** {doc.guide_type}")

        parts.append(f"\n{doc.content}")
        parts.append("---")

        return "\n".join(parts)
```

---

## 5. Embeddings Provider

### 5.1. `embeddings.py`

```python
"""Обгортка над LLM-провайдерами для генерації embeddings.

Використовує ті ж credentials, що й основний LLM провайдер ragnarbot.
Не потребує окремих API-ключів — повторно використовує існуючі.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Базовий інтерфейс embedding провайдера."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Вбудовує список документів. Повертає список векторів."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Вбудовує один запит. Повертає вектор."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Розмірність embedding вектора."""
        ...


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small / text-embedding-ada-002.

    Використовує credentials з ragnarbot config (openai API key або OAuth token).
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    # text-embedding-3-small: 1536 dims, $0.02/1M tokens
    # text-embedding-3-large: 3072 dims, $0.13/1M tokens
    # text-embedding-ada-002: 1536 dims, $0.10/1M tokens

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model or self.DEFAULT_MODEL
        self._dimensions = 1536  # default for small/ada

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # OpenAI підтримує batch embedding
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    @property
    def dimensions(self) -> int:
        return self._dimensions


class AnthropicVoyageEmbedding(EmbeddingProvider):
    """Anthropic рекомендує Voyage AI для embeddings.

    Якщо Voyage API key доступний — використовує його.
    Якщо ні — fallback до OpenAI embeddings через LiteLLM.

    voyage-3-large: 1024 dims
    voyage-3: 1024 dims
    voyage-code-3: 1024 dims
    """

    DEFAULT_MODEL = "voyage-3"

    def __init__(self, api_key: str, model: str | None = None):
        import httpx
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
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


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding для локальних моделей.

    nomic-embed-text: 768 dims
    mxbai-embed-large: 1024 dims
    all-minilm: 384 dims
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(self, base_url: str = "http://localhost:11434", model: str | None = None):
        import httpx
        self.model = model or self.DEFAULT_MODEL
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120.0)
        self._dimensions = 768  # nomic default

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:  # Ollama не підтримує batch
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


class LiteLLMEmbedding(EmbeddingProvider):
    """Fallback: використовує litellm для embed через будь-який провайдер.

    Працює з будь-якою моделлю, яку підтримує litellm.
    Повторно використовує existing API keys з ragnarbot config.
    """

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


def create_embedding_provider(
    provider: str | None,
    model: str | None,
    credentials: dict,
) -> EmbeddingProvider | None:
    """Фабрика embedding провайдера.

    Логіка вибору:
    1. Якщо provider вказано явно — використати його.
    2. Якщо ні — auto-detect з наявних credentials:
       - OpenAI API key → OpenAIEmbedding
       - Anthropic + Voyage key → AnthropicVoyageEmbedding
       - Ollama configured → OllamaEmbedding
       - Будь-який API key → LiteLLMEmbedding (fallback)
    3. Якщо нічого — повернути None (memory працює в files mode).
    """
    if provider == "openai" or (provider is None and credentials.get("openai_api_key")):
        api_key = credentials.get("openai_api_key")
        if api_key:
            return OpenAIEmbedding(api_key=api_key, model=model)

    if provider == "voyage" or (provider is None and credentials.get("voyage_api_key")):
        api_key = credentials.get("voyage_api_key")
        if api_key:
            return AnthropicVoyageEmbedding(api_key=api_key, model=model)

    if provider == "ollama" or (provider is None and credentials.get("ollama_base_url")):
        base_url = credentials.get("ollama_base_url", "http://localhost:11434")
        return OllamaEmbedding(base_url=base_url, model=model)

    # Fallback: litellm з будь-яким наявним ключем
    if credentials.get("openai_api_key") or credentials.get("anthropic_api_key"):
        return LiteLLMEmbedding(model=model or "text-embedding-3-small")

    logger.warning("No embedding provider available — memory will use files-only mode")
    return None
```

---

## 6. Document Chunker

### 6.1. `chunker.py`

```python
"""Розбиття тексту на чанки для embedding.

Адаптовано з PentAGI: RecursiveCharacterTextSplitter з підтримкою
кодових блоків та ієрархії заголовків.

Параметри за замовчуванням (з PentAGI):
- chunk_size: 2000 символів
- chunk_overlap: 100 символів
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    content: str
    index: int      # Порядковий номер чанка
    total: int      # Загальна кількість чанків


# Сепаратори в порядку пріоритету (від найкращого до fallback)
SEPARATORS = [
    "\n## ",        # Markdown H2
    "\n### ",       # Markdown H3
    "\n#### ",      # Markdown H4
    "\n\n",         # Подвійний перенос рядка (параграф)
    "\n",           # Одинарний перенос рядка
    ". ",           # Кінець речення
    ", ",           # Кома
    " ",            # Пробіл
    "",             # Символ за символом (крайній fallback)
]


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 100,
    preserve_code_blocks: bool = True,
) -> list[TextChunk]:
    """Рекурсивно розбиває текст на чанки.

    Args:
        text: Вхідний текст.
        chunk_size: Максимальний розмір чанка в символах.
        chunk_overlap: Кількість символів перекриття між чанками.
        preserve_code_blocks: Намагатися зберегти кодові блоки цілими.

    Returns:
        Список TextChunk з content, index, total.
    """
    if len(text) <= chunk_size:
        return [TextChunk(content=text, index=0, total=1)]

    # Якщо є кодові блоки — спочатку розділити навколо них
    if preserve_code_blocks:
        code_block_re = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        parts = _split_preserving_pattern(text, code_block_re, chunk_size)
        if parts:
            text_parts = parts
        else:
            text_parts = [text]
    else:
        text_parts = [text]

    # Рекурсивне розділення кожної частини
    chunks: list[str] = []
    for part in text_parts:
        if len(part) <= chunk_size:
            chunks.append(part)
        else:
            chunks.extend(_recursive_split(part, chunk_size, chunk_overlap, SEPARATORS))

    # Додати перекриття між чанками
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)

    total = len(chunks)
    return [TextChunk(content=c, index=i, total=total) for i, c in enumerate(chunks)]


def _recursive_split(
    text: str, chunk_size: int, overlap: int, separators: list[str]
) -> list[str]:
    """Рекурсивно розбиває текст, починаючи з найкращого сепаратора."""
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if not sep:  # Fallback до символів
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        # Зібрати чанки, об'єднуючи частини до chunk_size
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)

        # Рекурсивно розбити чанки, що все ще перевищують ліміт
        result: list[str] = []
        remaining_seps = separators[separators.index(sep) + 1:]
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                result.append(chunk)
            else:
                result.extend(_recursive_split(chunk, chunk_size, overlap, remaining_seps))

        return result

    return [text]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Додає перекриття: кожен чанк починається з останніх N символів попереднього."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + chunks[i])
    return result


def _split_preserving_pattern(
    text: str, pattern: re.Pattern, max_size: int
) -> list[str] | None:
    """Розбиває текст, зберігаючи блоки, що відповідають pattern, цілими."""
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    parts: list[str] = []
    last_end = 0
    for match in matches:
        if match.start() > last_end:
            parts.append(text[last_end:match.start()])
        parts.append(match.group())
        last_end = match.end()
    if last_end < len(text):
        parts.append(text[last_end:])

    return [p for p in parts if p.strip()]
```

---

## 7. Vector Backend

### 7.1. `backends/base.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class VectorBackend(ABC):
    """Абстракція над vector store.

    Реалізації: PgvectorBackend, ChromaBackend.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Ініціалізує бекенд (створює таблиці/колекції)."""
        ...

    @abstractmethod
    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> list[str]:
        """Зберігає документи. Повертає список ID."""
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[str, dict[str, str], float]]:
        """Пошук за similarity.

        Returns:
            Список (content, metadata, score) відсортований за score DESC.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        filters: Optional[dict[str, str]] = None,
        ids: Optional[list[str]] = None,
    ) -> int:
        """Видалення документів. Повертає кількість видалених."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Закриває з'єднання."""
        ...
```

### 7.2. `backends/pgvector_backend.py`

```python
"""PostgreSQL + pgvector backend.

Потребує:
- PostgreSQL 15+ з розширенням pgvector
- pip install asyncpg

SQL-схема створюється автоматично при initialize().
"""

from __future__ import annotations
from typing import Optional
import json
import uuid
import logging

logger = logging.getLogger(__name__)


# SQL для ініціалізації
INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector({dimensions}),
    metadata JSONB NOT NULL DEFAULT '{{}}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW індекс для швидкого cosine similarity пошуку
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN індекс для фільтрації за metadata
CREATE INDEX IF NOT EXISTS idx_memory_metadata
    ON memory_documents
    USING gin (metadata jsonb_path_ops);

-- Індекс для фільтрації за doc_type
CREATE INDEX IF NOT EXISTS idx_memory_doc_type
    ON memory_documents ((metadata->>'doc_type'));

-- Індекс за created_at для TTL / cleanup
CREATE INDEX IF NOT EXISTS idx_memory_created_at
    ON memory_documents (created_at);
"""

# SQL для пошуку
SEARCH_SQL = """
SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> $1::vector) AS score
FROM memory_documents
WHERE 1=1
    {filter_clauses}
    AND 1 - (embedding <=> $1::vector) >= $2
ORDER BY embedding <=> $1::vector
LIMIT $3
"""

# SQL для вставки
INSERT_SQL = """
INSERT INTO memory_documents (id, content, embedding, metadata)
VALUES ($1, $2, $3::vector, $4::jsonb)
"""

# SQL для видалення
DELETE_BY_FILTER_SQL = """
DELETE FROM memory_documents
WHERE 1=1 {filter_clauses}
"""

DELETE_BY_IDS_SQL = """
DELETE FROM memory_documents
WHERE id = ANY($1::uuid[])
"""


class PgvectorBackend:
    """pgvector backend implementation."""

    def __init__(self, database_url: str, dimensions: int):
        self.database_url = database_url
        self.dimensions = dimensions
        self._pool = None

    async def initialize(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(INIT_SQL.format(dimensions=self.dimensions))
        logger.info(f"pgvector backend initialized (dimensions={self.dimensions})")

    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> list[str]:
        ids = []
        async with self._pool.acquire() as conn:
            for text, emb, meta in zip(texts, embeddings, metadatas):
                doc_id = str(uuid.uuid4())
                embedding_str = "[" + ",".join(str(x) for x in emb) + "]"
                await conn.execute(
                    INSERT_SQL,
                    uuid.UUID(doc_id),
                    text,
                    embedding_str,
                    json.dumps(meta),
                )
                ids.append(doc_id)
        return ids

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[str, dict[str, str], float]]:
        filter_clauses = ""
        params_offset = 3  # $1=embedding, $2=threshold, $3=limit
        params = []

        if filters:
            clauses = []
            for i, (key, value) in enumerate(filters.items()):
                param_idx = params_offset + i + 1
                clauses.append(f"AND metadata->>'{key}' = ${param_idx}")
                params.append(value)
            filter_clauses = " ".join(clauses)

        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = SEARCH_SQL.format(filter_clauses=filter_clauses)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                embedding_str,
                score_threshold,
                k,
                *params,
            )

        results = []
        for row in rows:
            meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"])
            results.append((row["content"], meta, float(row["score"])))

        return results

    async def delete(
        self,
        filters: Optional[dict[str, str]] = None,
        ids: Optional[list[str]] = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            if ids:
                result = await conn.execute(
                    DELETE_BY_IDS_SQL,
                    [uuid.UUID(i) for i in ids],
                )
            elif filters:
                clauses = []
                params = []
                for i, (key, value) in enumerate(filters.items()):
                    clauses.append(f"AND metadata->>'{key}' = ${i + 1}")
                    params.append(value)
                sql = DELETE_BY_FILTER_SQL.format(filter_clauses=" ".join(clauses))
                result = await conn.execute(sql, *params)
            else:
                return 0

            # asyncpg returns "DELETE N"
            return int(result.split()[-1]) if result else 0

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
```

### 7.3. `backends/chroma_backend.py`

```python
"""ChromaDB backend — вбудований, без зовнішніх залежностей (крім pip install chromadb).

Зберігає дані локально у ~/.ragnarbot/chroma/.
Не потребує PostgreSQL.
Рекомендований для персонального використання.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)


class ChromaBackend:
    """ChromaDB local backend."""

    def __init__(self, persist_directory: str, collection_name: str = "ragnarbot_memory"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        import chromadb
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(f"ChromaDB initialized at {self.persist_directory}")

    async def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[str, dict[str, str], float]]:
        where = None
        if filters:
            if len(filters) == 1:
                key, val = next(iter(filters.items()))
                where = {key: val}
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB повертає cosine distance, не similarity
                # similarity = 1 - distance
                score = 1.0 - dist
                if score >= score_threshold:
                    output.append((doc, meta, score))

        return output

    async def delete(
        self,
        filters: Optional[dict[str, str]] = None,
        ids: Optional[list[str]] = None,
    ) -> int:
        if ids:
            self._collection.delete(ids=ids)
            return len(ids)
        elif filters:
            if len(filters) == 1:
                key, val = next(iter(filters.items()))
                where = {key: val}
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}
            # ChromaDB не повертає count — отримуємо спочатку
            existing = self._collection.get(where=where)
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                return len(existing["ids"])
        return 0

    async def close(self) -> None:
        pass  # ChromaDB PersistentClient авто-зберігає
```

---

## 8. VectorMemoryStore

### 8.1. `vector_store.py`

```python
"""VectorMemoryStore — високорівневий інтерфейс для семантичної пам'яті.

Об'єднує embedding provider + vector backend + chunker.
Реалізує двоступеневу стратегію пошуку з PentAGI (specific → global fallback).
"""

from __future__ import annotations
from typing import Optional
import logging

from .models import MemoryDocument, SearchResult, DocType
from .chunker import chunk_text
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorMemoryStore:
    """Семантична пам'ять з vector store backend."""

    def __init__(
        self,
        backend,              # VectorBackend (pgvector або chroma)
        embedder: EmbeddingProvider,
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
        default_threshold: float = 0.2,
        default_max_results: int = 5,
    ):
        self.backend = backend
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_threshold = default_threshold
        self.default_max_results = default_max_results
        self._initialized = False

    async def initialize(self) -> None:
        """Ініціалізує backend (створює таблиці/індекси)."""
        if not self._initialized:
            await self.backend.initialize()
            self._initialized = True

    # ─── STORE ──────────────────────────────────────────────────────

    async def store(self, doc: MemoryDocument) -> list[str]:
        """Зберігає документ у пам'яті.

        1. Розбиває content на чанки (якщо > chunk_size)
        2. Генерує embedding для кожного чанка
        3. Зберігає в vector backend з метаданими

        Returns:
            Список ID збережених чанків.
        """
        await self.initialize()

        total_size = len(doc.content.encode("utf-8"))

        # Розбиття на чанки
        chunks = chunk_text(
            doc.content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Підготовка текстів та метаданих
        texts = []
        metadatas = []
        for chunk in chunks:
            doc.chunk_index = chunk.index
            doc.chunk_total = chunk.total
            doc.total_size = total_size

            texts.append(chunk.content)
            metadatas.append(doc.to_metadata())

        # Embedding
        embeddings = await self.embedder.embed_documents(texts)

        # Зберігання
        ids = await self.backend.add_documents(texts, embeddings, metadatas)

        logger.info(
            f"Stored {len(ids)} chunks for {doc.doc_type.value} document "
            f"(total_size={total_size})"
        )
        return ids

    # ─── SEARCH ─────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        doc_type: Optional[DocType] = None,
        filters: Optional[dict[str, str]] = None,
        max_results: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """Семантичний пошук у пам'яті.

        Двоступенева стратегія (адаптовано з PentAGI):
        1. Пошук з усіма фільтрами
        2. Якщо 0 результатів і є специфічні фільтри — повторний пошук
           тільки з doc_type фільтром (глобальний fallback)

        Args:
            query: Текстовий запит (семантичний пошук)
            doc_type: Фільтр за типом документа
            filters: Додаткові фільтри (ключ → значення рядком)
            max_results: Ліміт результатів (default: self.default_max_results)
            threshold: Мінімальна similarity (default: self.default_threshold)

        Returns:
            Список SearchResult відсортований за score DESC.
        """
        await self.initialize()

        k = max_results or self.default_max_results
        t = threshold or self.default_threshold

        # Побудова фільтрів
        search_filters = dict(filters) if filters else {}
        if doc_type:
            search_filters["doc_type"] = doc_type.value

        # Embedding запиту
        query_embedding = await self.embedder.embed_query(query)

        # Крок 1: Пошук з усіма фільтрами
        raw_results = await self.backend.similarity_search(
            query_embedding=query_embedding,
            k=k,
            score_threshold=t,
            filters=search_filters if search_filters else None,
        )

        # Крок 2: Fallback — якщо 0 результатів і є специфічні фільтри
        has_specific_filters = bool(filters)  # є щось окрім doc_type
        if len(raw_results) == 0 and has_specific_filters:
            # Повторний пошук тільки з doc_type
            fallback_filters = {}
            if doc_type:
                fallback_filters["doc_type"] = doc_type.value

            raw_results = await self.backend.similarity_search(
                query_embedding=query_embedding,
                k=k,
                score_threshold=t,
                filters=fallback_filters if fallback_filters else None,
            )
            if raw_results:
                logger.info(
                    f"Fallback search found {len(raw_results)} results "
                    f"(specific filters returned 0)"
                )

        # Конвертація в SearchResult
        results = []
        for content, metadata, score in raw_results:
            doc = MemoryDocument.from_metadata(content, metadata)
            results.append(SearchResult(document=doc, score=score))

        return results

    # ─── DELETE ─────────────────────────────────────────────────────

    async def delete(
        self,
        doc_type: Optional[DocType] = None,
        filters: Optional[dict[str, str]] = None,
        ids: Optional[list[str]] = None,
    ) -> int:
        """Видалення документів з пам'яті."""
        await self.initialize()

        delete_filters = dict(filters) if filters else {}
        if doc_type:
            delete_filters["doc_type"] = doc_type.value

        count = await self.backend.delete(
            filters=delete_filters if delete_filters else None,
            ids=ids,
        )
        logger.info(f"Deleted {count} documents")
        return count

    # ─── CONVENIENCE ────────────────────────────────────────────────

    async def store_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        session_key: Optional[str] = None,
    ) -> list[str]:
        """Автоматичне збереження результату інструменту в пам'ять.

        Формат зберігання (адаптовано з PentAGI executor.go):
        - Аргументи інструменту як JSON
        - Результат інструменту

        Автоматично зберігаються результати від:
        exec, web_search, web_fetch, browser, spawn (subagent results)
        """
        import json as json_mod

        content = f"### Tool: {tool_name}\n\n"
        content += f"#### Arguments\n```json\n{json_mod.dumps(tool_args, indent=2, ensure_ascii=False)}\n```\n\n"
        content += f"#### Result\n{tool_result}"

        doc = MemoryDocument(
            content=content,
            doc_type=DocType.MEMORY,
            source_tool=tool_name,
            session_key=session_key,
        )
        return await self.store(doc)

    async def search_for_context(
        self,
        query: str,
        max_results: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """Пошук для автоматичної ін'єкції в system prompt.

        Повертає форматований текст для вставки в контекст.
        Обмежений за кількістю токенів (приблизно 4 символи = 1 токен).
        """
        results = await self.search(query, max_results=max_results)

        if not results:
            return ""

        parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Грубе наближення

        for i, result in enumerate(results):
            formatted = result.format_for_llm(i)
            if total_chars + len(formatted) > char_limit:
                break
            parts.append(formatted)
            total_chars += len(formatted)

        return "\n\n".join(parts)

    async def close(self) -> None:
        await self.backend.close()
```

---

## 9. Оновлений MemoryStore

### 9.1. `store.py` — оновлений головний клас

```python
"""MemoryStore — головний клас пам'яті ragnarbot.

Зберігає зворотну сумісність з існуючим файловим API
та додає семантичний пошук через VectorMemoryStore.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

from .models import MemoryDocument, SearchResult, DocType
from .vector_store import VectorMemoryStore

logger = logging.getLogger(__name__)


class MemoryStore:
    """Тришарова пам'ять ragnarbot.

    Layer 0: Файли (MEMORY.md, щоденні нотатки) — завжди доступний
    Layer 1: VectorMemoryStore (pgvector або ChromaDB) — якщо налаштований

    Зберігає повну зворотну сумісність з існуючим API.
    """

    def __init__(
        self,
        workspace: Path,
        vector_store: Optional[VectorMemoryStore] = None,
    ):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"

        # Layer 1: семантичний пошук (може бути None)
        self.vector: Optional[VectorMemoryStore] = vector_store

    @property
    def has_vector(self) -> bool:
        """Чи доступний семантичний пошук."""
        return self.vector is not None

    # ─── LAYER 0: FILE-BASED (існуючий API, без змін) ───────────────

    def get_today_file(self) -> Path:
        return self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def read_today(self) -> str:
        f = self.get_today_file()
        if f.exists():
            return f.read_text(encoding="utf-8")
        return ""

    def append_today(self, content: str) -> None:
        f = self.get_today_file()
        existing = f.read_text(encoding="utf-8") if f.exists() else ""
        separator = "\n\n" if existing.strip() else ""
        f.write_text(existing + separator + content, encoding="utf-8")

    def get_recent_memories(self, days: int = 7) -> str:
        cutoff = datetime.now() - timedelta(days=days)
        parts = []
        for f in sorted(self.memory_dir.glob("*.md"), reverse=True):
            if f.name == "MEMORY.md":
                continue
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                if file_date >= cutoff:
                    content = f.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(f"## {f.stem}\n\n{content}")
            except ValueError:
                continue
        return "\n\n".join(parts)

    def list_memory_files(self) -> list[Path]:
        files = sorted(self.memory_dir.glob("*.md"), reverse=True)
        return [f for f in files if f.name != "MEMORY.md"]

    def get_memory_context(self) -> str:
        """Повний контекст пам'яті для system prompt.

        ЗВОРОТНА СУМІСНІСТЬ: використовується якщо vector store недоступний
        або для доповнення семантичного пошуку.
        """
        parts = []
        long_term = self.read_long_term().strip()
        if long_term:
            parts.append(f"## Long-term Memory\n\n{long_term}")

        recent = self.get_recent_memories(days=7)
        if recent:
            parts.append(f"## Recent Notes\n\n{recent}")

        return "\n\n".join(parts)

    # ─── LAYER 1: SEMANTIC SEARCH (нове API) ─────────────────────────

    async def semantic_search(
        self,
        query: str,
        doc_type: Optional[DocType] = None,
        filters: Optional[dict[str, str]] = None,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Семантичний пошук у пам'яті.

        Якщо vector store недоступний — повертає порожній список.
        Агент повинен перевіряти has_vector перед викликом.
        """
        if not self.vector:
            return []
        return await self.vector.search(
            query=query,
            doc_type=doc_type,
            filters=filters,
            max_results=max_results,
        )

    async def semantic_store(self, doc: MemoryDocument) -> list[str]:
        """Зберігає документ у векторній пам'яті.

        Якщо vector store недоступний — тихо ігнорує (факт збережено у файлах).
        """
        if not self.vector:
            return []
        return await self.vector.store(doc)

    async def semantic_delete(
        self,
        doc_type: Optional[DocType] = None,
        filters: Optional[dict[str, str]] = None,
    ) -> int:
        """Видалення з векторної пам'яті."""
        if not self.vector:
            return 0
        return await self.vector.delete(doc_type=doc_type, filters=filters)

    async def store_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        session_key: Optional[str] = None,
    ) -> None:
        """Автоматично зберігає результат інструменту.

        Викликається з agent loop після виконання інструменту.
        Зберігає тільки якщо vector store доступний.
        """
        if self.vector:
            await self.vector.store_tool_result(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                session_key=session_key,
            )

    # ─── CONTEXT BUILDING (нова логіка) ──────────────────────────────

    async def get_context_for_query(
        self,
        query: str,
        max_results: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """Контекст пам'яті, адаптований під запит.

        Якщо vector store доступний:
          - Семантичний пошук по запиту → top-K релевантних фактів
          - + Long-term memory (завжди)

        Якщо ні:
          - get_memory_context() як зараз
        """
        parts = []

        # Завжди: long-term memory (файл)
        long_term = self.read_long_term().strip()
        if long_term:
            parts.append(f"## Long-term Memory\n\n{long_term}")

        # Семантичний пошук (якщо доступний)
        if self.vector:
            semantic = await self.vector.search_for_context(
                query=query,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            if semantic:
                parts.append(f"## Relevant Memory\n\n{semantic}")
        else:
            # Fallback: щоденні нотатки
            recent = self.get_recent_memories(days=7)
            if recent:
                parts.append(f"## Recent Notes\n\n{recent}")

        return "\n\n".join(parts)

    # ─── LIFECYCLE ───────────────────────────────────────────────────

    async def close(self) -> None:
        if self.vector:
            await self.vector.close()
```

---

## 10. LLM Tools для пам'яті

### 10.1. `tools.py`

```python
"""LLM інструменти для роботи з пам'яттю.

Додати в реєстр інструментів ragnarbot.
Агент зможе явно зберігати та шукати у пам'яті.
"""

# Tool definitions (JSON Schema для function calling)

MEMORY_SEARCH_TOOL = {
    "name": "memory_search",
    "description": (
        "Search your semantic long-term memory for relevant information. "
        "Use this when you need to recall past conversations, saved guides, "
        "code snippets, or answers from previous research. "
        "The search is semantic — describe WHAT you're looking for, not exact keywords."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A semantically rich search query describing what you want to find. "
                    "Be specific and descriptive. "
                    "Example: 'How to configure nginx reverse proxy with SSL' "
                    "rather than just 'nginx'."
                ),
            },
            "doc_type": {
                "type": "string",
                "enum": ["memory", "guide", "answer", "code", "conversation"],
                "description": (
                    "Filter by document type. "
                    "memory=general facts, guide=how-to instructions, "
                    "answer=research results, code=code snippets, "
                    "conversation=past chat excerpts. "
                    "Omit to search all types."
                ),
            },
            "max_results": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of results (1-10).",
            },
        },
        "required": ["query"],
    },
}

MEMORY_STORE_TOOL = {
    "name": "memory_store",
    "description": (
        "Save important information to your semantic long-term memory. "
        "Use this to remember facts, guides, code snippets, or research results "
        "that will be useful in future conversations. "
        "Information stored here can be retrieved later with memory_search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The content to store. Write in clear Markdown. "
                    "Include enough context so the information is useful standalone."
                ),
            },
            "doc_type": {
                "type": "string",
                "enum": ["memory", "guide", "answer", "code", "conversation"],
                "default": "memory",
                "description": (
                    "Type of document. "
                    "memory=general fact/observation, "
                    "guide=step-by-step instructions, "
                    "answer=research result/analysis, "
                    "code=code snippet with explanation, "
                    "conversation=important conversation excerpt."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "The question this content answers. "
                    "Helps with retrieval later. "
                    "Example: 'How to deploy Django app to production?'"
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization (e.g., ['python', 'deployment', 'django']).",
            },
            "code_lang": {
                "type": "string",
                "description": "Programming language (only for doc_type=code).",
            },
        },
        "required": ["content"],
    },
}

MEMORY_FORGET_TOOL = {
    "name": "memory_forget",
    "description": (
        "Delete specific memories from semantic memory. "
        "Use when the user asks to forget something or when information is outdated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find memories to delete.",
            },
            "doc_type": {
                "type": "string",
                "enum": ["memory", "guide", "answer", "code", "conversation"],
                "description": "Filter by document type.",
            },
            "confirm_count": {
                "type": "integer",
                "description": (
                    "Maximum number of documents to delete. "
                    "Safety limit to prevent accidental mass deletion."
                ),
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# Tool handler implementations

async def handle_memory_search(memory_store, args: dict) -> str:
    """Handler для memory_search tool."""
    query = args["query"]
    doc_type_str = args.get("doc_type")
    max_results = min(args.get("max_results", 5), 10)

    doc_type = DocType(doc_type_str) if doc_type_str else None

    results = await memory_store.semantic_search(
        query=query,
        doc_type=doc_type,
        max_results=max_results,
    )

    if not results:
        return "No relevant memories found for this query."

    parts = [f"Found {len(results)} relevant memories:\n"]
    for i, result in enumerate(results):
        parts.append(result.format_for_llm(i))

    return "\n\n".join(parts)


async def handle_memory_store(memory_store, args: dict) -> str:
    """Handler для memory_store tool."""
    from .models import MemoryDocument, DocType

    content = args["content"]
    doc_type = DocType(args.get("doc_type", "memory"))

    doc = MemoryDocument(
        content=content,
        doc_type=doc_type,
        question=args.get("question"),
        tags=args.get("tags", []),
        code_lang=args.get("code_lang"),
    )

    ids = await memory_store.semantic_store(doc)

    if ids:
        return f"Stored in memory ({len(ids)} chunks, type={doc_type.value})."
    else:
        return "Memory stored in file-based memory (vector store not available)."


async def handle_memory_forget(memory_store, args: dict) -> str:
    """Handler для memory_forget tool."""
    from .models import DocType

    query = args["query"]
    doc_type_str = args.get("doc_type")
    confirm_count = args.get("confirm_count", 5)

    doc_type = DocType(doc_type_str) if doc_type_str else None

    # Спочатку знайти що видалятимемо
    results = await memory_store.semantic_search(
        query=query,
        doc_type=doc_type,
        max_results=confirm_count,
    )

    if not results:
        return "No memories found matching this query. Nothing to delete."

    # Видалити знайдене
    # Тут потрібен доступ до ID — додамо в SearchResult пізніше
    # Поки використовуємо фільтри
    count = await memory_store.semantic_delete(doc_type=doc_type)

    return f"Deleted {count} memories matching the query."
```

---

## 11. Інтеграція в існуючий код

### 11.1. Зміни в `agent/context.py`

```python
# БУЛО:
memory = self.memory.get_memory_context()
if memory:
    parts.append(f"# Memory\n\n{memory}")

# СТАЛО:
async def build_system_prompt(self, user_message: str = "", ...):
    # ... existing code ...

    # Memory section (НОВЕ: семантичний пошук за запитом)
    if user_message and self.memory.has_vector:
        memory = await self.memory.get_context_for_query(
            query=user_message,
            max_results=self.memory_config.auto_inject_max_results,
            max_tokens=self.memory_config.auto_inject_max_tokens,
        )
    else:
        memory = self.memory.get_memory_context()  # fallback до файлів

    if memory:
        parts.append(f"# Memory\n\n{memory}")
```

### 11.2. Зміни в `agent/loop.py`

```python
# В _process_tool_call(), після виконання інструменту:

# Автоматичне збереження результатів інструментів у пам'ять
TOOLS_TO_STORE = {"exec", "web_search", "web_fetch", "browser", "spawn"}

if tool_name in TOOLS_TO_STORE and self.memory.has_vector:
    await self.memory.store_tool_result(
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        session_key=session.key,
    )
```

### 11.3. Зміни в `agent/tools/registry.py`

```python
# Додати нові інструменти до реєстру
from ragnarbot.agent.memory.tools import (
    MEMORY_SEARCH_TOOL,
    MEMORY_STORE_TOOL,
    MEMORY_FORGET_TOOL,
    handle_memory_search,
    handle_memory_store,
    handle_memory_forget,
)

# В register_tools():
if memory_store.has_vector:
    registry.register("memory_search", MEMORY_SEARCH_TOOL,
                       lambda args: handle_memory_search(memory_store, args))
    registry.register("memory_store", MEMORY_STORE_TOOL,
                       lambda args: handle_memory_store(memory_store, args))
    registry.register("memory_forget", MEMORY_FORGET_TOOL,
                       lambda args: handle_memory_forget(memory_store, args))
```

### 11.4. Ініціалізація в `cli/commands.py` або `gateway`

```python
from ragnarbot.agent.memory.store import MemoryStore
from ragnarbot.agent.memory.vector_store import VectorMemoryStore
from ragnarbot.agent.memory.embeddings import create_embedding_provider
from ragnarbot.agent.memory.backends.chroma_backend import ChromaBackend
from ragnarbot.agent.memory.backends.pgvector_backend import PgvectorBackend

async def create_memory_store(config, credentials, workspace) -> MemoryStore:
    """Фабрика MemoryStore з конфігурації."""

    vector_store = None

    if config.memory.backend != "files":
        # Створити embedding provider
        embedder = create_embedding_provider(
            provider=config.memory.embedding_provider,
            model=config.memory.embedding_model,
            credentials=credentials,
        )

        if embedder:
            # Створити backend
            if config.memory.backend == "pgvector":
                backend = PgvectorBackend(
                    database_url=config.memory.database_url,
                    dimensions=embedder.dimensions,
                )
            elif config.memory.backend == "chroma":
                chroma_path = config.memory.chroma_path or str(workspace.parent / "chroma")
                backend = ChromaBackend(persist_directory=chroma_path)

            vector_store = VectorMemoryStore(
                backend=backend,
                embedder=embedder,
                chunk_size=config.memory.chunk_size,
                chunk_overlap=config.memory.chunk_overlap,
                default_threshold=config.memory.similarity_threshold,
                default_max_results=config.memory.max_results,
            )

    return MemoryStore(workspace=workspace, vector_store=vector_store)
```

---

## 12. Залежності

### 12.1. Додати в `pyproject.toml`

```toml
[project.optional-dependencies]
# Семантична пам'ять — ChromaDB (рекомендовано для персонального використання)
memory-chroma = ["chromadb>=0.5.0"]

# Семантична пам'ять — pgvector (для production / shared використання)
memory-pgvector = ["asyncpg>=0.29.0"]

# Voyage AI embeddings (рекомендовано для Anthropic users)
embeddings-voyage = ["httpx>=0.25.0"]  # вже є в основних залежностях
```

Жодних обов'язкових нових залежностей. ChromaDB або asyncpg — опціональні.

---

## 13. Міграція існуючої пам'яті

### 13.1. Одноразовий імпорт файлової пам'яті в vector store

```python
async def migrate_file_memory_to_vector(memory_store: MemoryStore) -> int:
    """Імпортує існуючу файлову пам'ять у vector store.

    Викликається один раз при першому включенні vector backend.
    """
    if not memory_store.has_vector:
        return 0

    count = 0

    # Імпорт MEMORY.md
    long_term = memory_store.read_long_term()
    if long_term.strip():
        doc = MemoryDocument(
            content=long_term,
            doc_type=DocType.MEMORY,
            tags=["imported", "long-term"],
        )
        await memory_store.semantic_store(doc)
        count += 1

    # Імпорт щоденних нотаток
    for f in memory_store.list_memory_files():
        content = f.read_text(encoding="utf-8").strip()
        if content:
            doc = MemoryDocument(
                content=content,
                doc_type=DocType.MEMORY,
                tags=["imported", "daily-note", f.stem],
            )
            await memory_store.semantic_store(doc)
            count += 1

    return count
```

---

## 14. Порядок реалізації

1. **`models.py`** — dataclasses (немає залежностей)
2. **`chunker.py`** — текстовий спліттер (немає залежностей)
3. **`embeddings.py`** — embedding providers (використовує існуючі LLM clients)
4. **`backends/base.py`** — абстракція
5. **`backends/chroma_backend.py`** — ChromaDB (рекомендовано першим — простіше тестувати)
6. **`backends/pgvector_backend.py`** — pgvector (другий пріоритет)
7. **`vector_store.py`** — високорівневий інтерфейс
8. **`store.py`** — оновлений MemoryStore (зворотна сумісність!)
9. **`tools.py`** — LLM інструменти
10. **Інтеграція** — зміни в context.py, loop.py, registry.py, commands.py
11. **Міграція** — імпорт існуючої пам'яті
12. **Тести**
