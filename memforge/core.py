"""MemForge: Main entry point.

Usage:
    # With preset (easiest)
    mf = MemForge(preset="openrouter")
    mf = MemForge(preset="anthropic")
    mf = MemForge(preset="google")
    mf = MemForge(preset="ollama")

    # With explicit config
    from memforge.providers import ProviderConfig
    mf = MemForge(
        llm_config=ProviderConfig(provider="openrouter", model="anthropic/claude-sonnet-4-6"),
        embedding_config=ProviderConfig(provider="openrouter", model="openai/text-embedding-3-small"),
    )

    # Use it
    mf.add([{"role": "user", "content": "I prefer dark mode"}])
    results = mf.search("user preferences")
"""

from __future__ import annotations

import logging

from memforge.ingestion.fact_extractor import FactExtractor
from memforge.ingestion.pipeline import IngestionPipeline
from memforge.models import AtomicFact, Message, ScoredFact
from memforge.providers import (
    BaseEmbeddingClient,
    BaseLLMClient,
    ProviderConfig,
    create_embedding,
    create_llm,
    get_preset,
)
from memforge.retrieval.engine import RetrievalEngine
from memforge.storage.engine import StorageEngine

logger = logging.getLogger(__name__)


class MemForge:
    """High-level API for MemForge memory system."""

    def __init__(
        self,
        db_path: str = "~/.memforge/memforge.db",
        # Option 1: Use a preset
        preset: str | None = None,
        # Option 2: Explicit provider configs
        llm_config: ProviderConfig | None = None,
        embedding_config: ProviderConfig | None = None,
        # Option 3: Pre-built clients (for advanced use / testing)
        llm: BaseLLMClient | None = None,
        embedding: BaseEmbeddingClient | None = None,
        # Embedding dimensions (for vector index)
        embedding_dim: int = 768,
    ):
        # Resolve provider configs
        if preset:
            preset_configs = get_preset(preset)
            llm_config = llm_config or preset_configs["llm"]
            embedding_config = embedding_config or preset_configs["embedding"]

        # Build clients
        if llm is None:
            if llm_config is None:
                llm_config = ProviderConfig(provider="openai", model="gpt-4.1-mini")
            llm = create_llm(llm_config)

        if embedding is None:
            if embedding_config is None:
                embedding_config = ProviderConfig(provider="openai", model="text-embedding-3-small")
            embedding = create_embedding(embedding_config, dimensions=embedding_dim)

        self._llm = llm
        self._embedding = embedding

        # Storage
        self.storage = StorageEngine(db_path)
        self.storage.connect()

        # Ingestion
        extractor = FactExtractor(llm=llm)
        self.ingestion = IngestionPipeline(
            storage=self.storage,
            extractor=extractor,
            embedding=embedding,
        )

        # Retrieval
        self.retrieval = RetrievalEngine(
            storage=self.storage,
            embedding=embedding,
        )

    def add(
        self,
        messages: list[dict[str, str]] | list[Message],
        session_id: str | None = None,
        observation_date: str | None = None,
    ) -> list[AtomicFact]:
        """Ingest conversation messages into memory.

        Args:
            messages: List of {role, content} dicts or Message objects.
            session_id: Optional session identifier.
            observation_date: ISO 8601 date for temporal normalization anchor.

        Returns:
            List of stored AtomicFact objects.
        """
        msgs = []
        for m in messages:
            if isinstance(m, Message):
                msgs.append(m)
            elif isinstance(m, dict):
                msgs.append(Message(role=m["role"], content=m["content"]))
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")

        return self.ingestion.ingest(
            messages=msgs,
            session_id=session_id,
            observation_date=observation_date,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        format: str = "full",
    ) -> list[ScoredFact]:
        """Search memories by natural language query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            format: "full" for complete facts, "l0" for token-efficient abstracts.

        Returns:
            List of ScoredFact objects ranked by relevance.
        """
        return self.retrieval.search(query=query, top_k=top_k, format=format)

    def get_context(self, query: str, top_k: int = 5) -> str:
        """Get formatted memory context for LLM injection.

        Returns a string ready to be prepended to system prompt.
        """
        results = self.search(query, top_k=top_k, format="full")
        if not results:
            return ""

        lines = ["## Relevant Memories"]
        for i, sf in enumerate(results, 1):
            lines.append(f"{i}. [{sf.fact.fact_type.value}] {sf.fact.content}")
            if sf.fact.entities:
                lines.append(f"   Entities: {', '.join(sf.fact.entities)}")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "total_facts": self.storage.count_facts(),
            "db_path": str(self.storage.db_path),
        }

    def close(self) -> None:
        self.storage.close()
