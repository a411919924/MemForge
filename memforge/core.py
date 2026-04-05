"""MemForge: Main entry point."""

from __future__ import annotations

import logging
from datetime import datetime

from memforge.ingestion.fact_extractor import CloudFactExtractor
from memforge.ingestion.pipeline import IngestionPipeline
from memforge.models import AtomicFact, Message, ScoredFact
from memforge.retrieval.engine import RetrievalEngine
from memforge.storage.engine import StorageEngine

logger = logging.getLogger(__name__)


class MemForge:
    """High-level API for MemForge memory system."""

    def __init__(
        self,
        db_path: str = "~/.memforge/memforge.db",
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 768,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        # Shared clients
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._openai = OpenAI(**client_kwargs)

        # Storage
        self.storage = StorageEngine(db_path)
        self.storage.connect()

        # Ingestion
        extractor = CloudFactExtractor(model=llm_model, api_key=api_key, base_url=base_url)
        self.ingestion = IngestionPipeline(
            storage=self.storage,
            extractor=extractor,
            embedding_client=self._openai,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        # Retrieval
        self.retrieval = RetrievalEngine(
            storage=self.storage,
            embedding_client=self._openai,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
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
        # Normalize to Message objects
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
