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

        # Direct score fusion (ReMe-style)
        fused = self._score_fusion(semantic_results, bm25_results)

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

    def _score_fusion(
        self,
        semantic_results: list[ScoredFact],
        bm25_results: list[ScoredFact],
    ) -> list[ScoredFact]:
        """Direct score fusion: score = w_sem * sim + w_bm25 * bm25_score."""
        scores: dict[str, float] = defaultdict(float)
        facts: dict[str, ScoredFact] = {}

        for sf in semantic_results:
            fid = sf.fact.id
            scores[fid] += self.semantic_weight * sf.score
            facts[fid] = sf

        for sf in bm25_results:
            fid = sf.fact.id
            scores[fid] += self.bm25_weight * sf.score
            if fid not in facts:
                facts[fid] = sf

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for fid, score in ranked:
            sf = facts[fid]
            sf.score = score
            results.append(sf)
        return results
