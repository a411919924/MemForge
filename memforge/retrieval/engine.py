"""Hybrid retrieval engine: semantic + BM25 fusion."""

from __future__ import annotations

import logging
from collections import defaultdict

from memforge.models import ScoredFact
from memforge.providers import BaseEmbeddingClient
from memforge.storage.engine import StorageEngine

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Two-channel hybrid retrieval: semantic + BM25 with fixed-weight RRF."""

    def __init__(
        self,
        storage: StorageEngine,
        embedding: BaseEmbeddingClient,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.storage = storage
        self.embedding = embedding
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 20,
        format: str = "full",
    ) -> list[ScoredFact]:
        """Hybrid search: semantic + BM25, fixed-weight RRF fusion."""
        # Channel 1: Semantic search
        query_embedding = self._embed_query(query)
        semantic_results = self.storage.search_vector(query_embedding, limit=top_k * 2) if query_embedding else []

        # Channel 2: BM25 full-text search
        bm25_results = self.storage.search_bm25(query, limit=top_k)

        # RRF fusion
        fused = self._weighted_rrf(
            channels=[semantic_results, bm25_results],
            weights=[self.semantic_weight, self.bm25_weight],
            k=60,
        )

        results = fused[:top_k]
        if format == "l0":
            for r in results:
                if r.fact.l0_abstract:
                    r.fact.content = r.fact.l0_abstract
        return results

    def _embed_query(self, query: str) -> list[float] | None:
        try:
            return self.embedding.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    @staticmethod
    def _weighted_rrf(
        channels: list[list[ScoredFact]],
        weights: list[float],
        k: int = 60,
    ) -> list[ScoredFact]:
        """Weighted Reciprocal Rank Fusion."""
        scores: dict[str, float] = defaultdict(float)
        facts: dict[str, ScoredFact] = {}
        channel_names = ["semantic", "bm25"]

        for channel_idx, (channel_results, weight) in enumerate(zip(channels, weights)):
            if weight == 0:
                continue
            for rank, scored_fact in enumerate(channel_results):
                fid = scored_fact.fact.id
                rrf_score = weight / (k + rank + 1)
                scores[fid] += rrf_score
                if fid not in facts:
                    facts[fid] = scored_fact

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for fid, score in ranked:
            sf = facts[fid]
            sf.score = score
            results.append(sf)
        return results
