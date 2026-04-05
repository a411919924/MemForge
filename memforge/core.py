"""MemForge: Main entry point.

Usage:
    # From YAML config (recommended)
    mf = MemForge()                            # auto-finds ./memforge.yaml or ~/.memforge/config.yaml
    mf = MemForge(config_path="my-config.yaml") # explicit path

    # With preset
    mf = MemForge(preset="openrouter")

    # With explicit provider configs
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
from pathlib import Path

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
    """High-level API for MemForge memory system.

    Config resolution order (first match wins):
    1. Pre-built clients (llm=, embedding=)
    2. Explicit provider configs (llm_config=, embedding_config=)
    3. Preset name (preset=)
    4. YAML config file (config_path= or auto-discovered)
    """

    def __init__(
        self,
        db_path: str | None = None,
        # Option 1: YAML config file
        config_path: str | None = None,
        # Option 2: Use a preset
        preset: str | None = None,
        # Option 3: Explicit provider configs
        llm_config: ProviderConfig | None = None,
        embedding_config: ProviderConfig | None = None,
        # Option 4: Pre-built clients (for advanced use / testing)
        llm: BaseLLMClient | None = None,
        embedding: BaseEmbeddingClient | None = None,
        # Embedding dimensions (for vector index)
        embedding_dim: int | None = None,
    ):
        # Try loading YAML config as base (lowest priority, overridden by explicit args)
        yaml_config = None
        if config_path or (llm_config is None and preset is None and llm is None):
            try:
                from memforge.config import load_config
                yaml_config = load_config(config_path)
            except FileNotFoundError:
                if config_path:
                    raise  # Explicit path must exist
                # Auto-discovery failed, that's fine
            except ImportError:
                pass  # pyyaml not installed

        # Resolve db_path
        if db_path is None:
            db_path = yaml_config.storage_path if yaml_config else "~/.memforge/memforge.db"

        # Resolve embedding_dim
        if embedding_dim is None:
            embedding_dim = yaml_config.embedding_dim if yaml_config else 768

        # Resolve provider configs (explicit > preset > yaml > defaults)
        ingestion_llm_config = llm_config
        if preset:
            preset_configs = get_preset(preset)
            ingestion_llm_config = ingestion_llm_config or preset_configs["llm"]
            embedding_config = embedding_config or preset_configs["embedding"]
        elif yaml_config:
            ingestion_llm_config = ingestion_llm_config or yaml_config.ingestion_llm
            embedding_config = embedding_config or yaml_config.embedding

        # Build clients
        if llm is None:
            if ingestion_llm_config is None:
                ingestion_llm_config = ProviderConfig(provider="openai", model="gpt-4.1-mini")
            llm = create_llm(ingestion_llm_config)

        if embedding is None:
            if embedding_config is None:
                embedding_config = ProviderConfig(provider="openai", model="text-embedding-3-small")
            embedding = create_embedding(embedding_config, dimensions=embedding_dim)

        self._llm = llm
        self._embedding = embedding
        self._config = yaml_config

        # Storage
        self.storage = StorageEngine(db_path, embedding_dim=embedding_dim)
        self.storage.connect()

        # Ingestion params from yaml config
        ing_kwargs = {}
        if yaml_config:
            ing_kwargs["window_size"] = yaml_config.ingestion.window_size
            ing_kwargs["window_overlap"] = yaml_config.ingestion.window_overlap
            ing_kwargs["dedup_threshold"] = yaml_config.ingestion.dedup_threshold

        # Ingestion
        extractor = FactExtractor(llm=llm)
        self.ingestion = IngestionPipeline(
            storage=self.storage,
            extractor=extractor,
            embedding=embedding,
            **ing_kwargs,
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
