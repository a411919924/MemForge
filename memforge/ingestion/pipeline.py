"""Ingestion pipeline orchestrator — the core of MemForge.

Implements the full write pipeline:
1. Sliding window segmentation
2. Fact extraction (Mode A/B/C)
3. Deduplication & conflict detection
4. L0 summary generation
5. Embedding generation
6. Graph edge construction
7. Storage
"""

from __future__ import annotations

import logging
from datetime import datetime

from memforge.ingestion.fact_extractor import FactExtractor
from memforge.models import AtomicFact, ConflictAction, ConflictResult, Message
from memforge.providers import BaseEmbeddingClient
from memforge.storage.engine import StorageEngine

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline."""

    def __init__(
        self,
        storage: StorageEngine,
        extractor: FactExtractor,
        embedding: BaseEmbeddingClient,
        window_size: int = 40,
        window_overlap: int = 2,
        dedup_threshold: float = 0.85,
    ):
        self.storage = storage
        self.extractor = extractor
        self.embedding = embedding
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.dedup_threshold = dedup_threshold

    def ingest(
        self,
        messages: list[Message],
        session_id: str | None = None,
        observation_date: str | None = None,
    ) -> list[AtomicFact]:
        """Full ingestion pipeline: messages → stored atomic facts."""
        if not messages:
            return []

        if observation_date is None:
            observation_date = datetime.utcnow().strftime("%Y-%m-%d")

        logger.info(f"Ingesting {len(messages)} messages (session={session_id})")

        # Step 1: Sliding window segmentation
        windows = self._segment_windows(messages)
        logger.info(f"Split into {len(windows)} windows")

        # Step 2: Extract facts from each window (parallelized)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_window(i: int, window: list[Message]) -> tuple[int, list[AtomicFact]]:
            logger.info(f"Extracting facts from window {i+1}/{len(windows)}")
            facts = self.extractor.extract(
                messages=window,
                prev_facts=None,  # No cross-window dedup — handled by dedup stage
                observation_date=observation_date,
            )
            return i, facts

        from tqdm import tqdm

        all_facts: list[AtomicFact] = []
        with ThreadPoolExecutor(max_workers=min(8, len(windows))) as executor:
            futures = [executor.submit(_extract_window, i, w) for i, w in enumerate(windows)]
            window_results: list[tuple[int, list[AtomicFact]]] = []
            with tqdm(total=len(windows), desc="Extracting facts", unit="win") as pbar:
                for future in as_completed(futures):
                    window_results.append(future.result())
                    pbar.update(1)

        # Merge in original window order
        window_results.sort(key=lambda x: x[0])
        for _, facts in window_results:
            all_facts.extend(facts)

        logger.info(f"Extracted {len(all_facts)} raw facts")

        if not all_facts:
            return []

        # Step 3: Generate L0 summaries
        logger.info("Generating L0 summaries")
        all_facts = self.extractor.generate_l0(all_facts)

        # Step 4: Generate embeddings
        logger.info("Generating embeddings")
        all_facts = self._embed_facts(all_facts)

        # Step 5: Dedup & conflict detection
        logger.info("Running deduplication")
        accepted_facts = self._dedup_and_resolve(all_facts)
        logger.info(f"Accepted {len(accepted_facts)}/{len(all_facts)} facts after dedup")

        # Step 6: Set session info and store
        for fact in accepted_facts:
            fact.source_session = session_id

        self.storage.insert_facts_batch(accepted_facts)

        # Step 7: Build graph edges
        logger.info("Building graph edges")
        self._build_graph_edges(accepted_facts)

        logger.info(f"Ingestion complete: {len(accepted_facts)} facts stored")
        return accepted_facts

    def _segment_windows(self, messages: list[Message]) -> list[list[Message]]:
        """Split messages into overlapping sliding windows."""
        if len(messages) <= self.window_size:
            return [messages]

        windows = []
        step = self.window_size - self.window_overlap
        for start in range(0, len(messages), step):
            end = min(start + self.window_size, len(messages))
            windows.append(messages[start:end])
            if end >= len(messages):
                break
        return windows

    def _embed_facts(self, facts: list[AtomicFact]) -> list[AtomicFact]:
        """Generate embeddings for all facts in batch."""
        texts = [f.content for f in facts]
        if not texts:
            return facts

        try:
            embeddings = self.embedding.embed(texts)
            for fact, emb in zip(facts, embeddings):
                fact.embedding = emb
        except Exception as e:
            logger.error(f"Embedding failed: {e}")

        return facts

    def _dedup_and_resolve(self, facts: list[AtomicFact]) -> list[AtomicFact]:
        """Three-layer deduplication: hash → semantic → accept."""
        accepted = []
        for fact in facts:
            result = self._check_conflict(fact)
            if result.action == ConflictAction.ADD:
                accepted.append(fact)
            elif result.action == ConflictAction.UPDATE:
                # Update existing fact with new content
                if result.existing_fact:
                    fact.id = result.existing_fact.id  # Keep same ID
                    accepted.append(fact)
            elif result.action == ConflictAction.SUPERSEDE:
                # New fact supersedes old — store both, mark edge
                accepted.append(fact)
                if result.existing_fact:
                    self.storage.insert_edge(
                        fact.id, result.existing_fact.id, "superseded_by"
                    )
            # NOOP: skip duplicate
        return accepted

    def _check_conflict(self, candidate: AtomicFact) -> ConflictResult:
        """Check if a candidate fact conflicts with existing facts."""
        # Layer 1: Exact hash match
        existing_id = self.storage.check_hash_exists(candidate.content_hash)
        if existing_id:
            existing = self.storage.get_fact(existing_id)
            return ConflictResult(
                action=ConflictAction.NOOP,
                candidate=candidate,
                existing_fact=existing,
                similarity=1.0,
            )

        # Layer 2: Semantic similarity (via vector search)
        if candidate.embedding is not None:
            similar = self.storage.search_vector(candidate.embedding, limit=3)
            for scored in similar:
                if scored.score >= self.dedup_threshold:
                    # High similarity — likely duplicate or update
                    if scored.score >= 0.95:
                        return ConflictResult(
                            action=ConflictAction.NOOP,
                            candidate=candidate,
                            existing_fact=scored.fact,
                            similarity=scored.score,
                        )
                    else:
                        # Similar but different enough — update
                        return ConflictResult(
                            action=ConflictAction.UPDATE,
                            candidate=candidate,
                            existing_fact=scored.fact,
                            similarity=scored.score,
                        )

        # No conflict found — add as new
        return ConflictResult(action=ConflictAction.ADD, candidate=candidate)

    def _build_graph_edges(self, facts: list[AtomicFact]) -> None:
        """Build entity and semantic graph edges between facts."""
        # Entity co-occurrence edges
        entity_index: dict[str, list[str]] = {}
        for fact in facts:
            for entity in fact.entities:
                entity_lower = entity.lower()
                if entity_lower not in entity_index:
                    entity_index[entity_lower] = []
                entity_index[entity_lower].append(fact.id)

        for entity, fact_ids in entity_index.items():
            if len(fact_ids) < 2:
                continue
            # Connect all facts sharing an entity (limit fan-out)
            for i, fid1 in enumerate(fact_ids[:10]):
                for fid2 in fact_ids[i + 1:10]:
                    self.storage.insert_edge(fid1, fid2, "entity_shared")

        # Temporal adjacency edges
        temporal_facts = [f for f in facts if f.temporal.referenced_date]
        temporal_facts.sort(key=lambda f: f.temporal.referenced_date or "")
        for i in range(len(temporal_facts) - 1):
            self.storage.insert_edge(
                temporal_facts[i].id, temporal_facts[i + 1].id, "temporal_adjacent"
            )
