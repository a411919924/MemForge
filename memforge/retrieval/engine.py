"""Multi-channel retrieval engine with intent-adaptive fusion."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from memforge.models import ScoredFact
from memforge.providers import BaseEmbeddingClient
from memforge.storage.engine import StorageEngine

logger = logging.getLogger(__name__)


class IntentType:
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    MULTI_HOP = "multi_hop"
    OPINION = "opinion"
    VAGUE = "vague"


# Intent-adaptive channel weights
INTENT_WEIGHTS = {
    #                    semantic  bm25   graph  temporal
    IntentType.FACTUAL:   [0.55,  0.35,  0.05,  0.05],
    IntentType.TEMPORAL:  [0.30,  0.25,  0.05,  0.40],
    IntentType.ENTITY:    [0.40,  0.20,  0.30,  0.10],
    IntentType.MULTI_HOP: [0.35,  0.15,  0.35,  0.15],
    IntentType.OPINION:   [0.55,  0.35,  0.10,  0.00],
    IntentType.VAGUE:     [0.45,  0.30,  0.15,  0.10],
}


class RetrievalEngine:
    """Four-channel hybrid retrieval with intent-adaptive weighted RRF."""

    def __init__(
        self,
        storage: StorageEngine,
        embedding: BaseEmbeddingClient,
    ):
        self.storage = storage
        self.embedding = embedding

    def search(
        self,
        query: str,
        top_k: int = 10,
        format: str = "full",  # "full" | "l0"
    ) -> list[ScoredFact]:
        """Multi-channel search with intent-adaptive fusion."""
        # Step 1: Classify intent
        intent = self._classify_intent(query)
        weights = INTENT_WEIGHTS[intent]
        logger.info(f"Query intent: {intent}, weights: {weights}")

        # Step 2: Run channels (could be async, keeping sync for baseline simplicity)
        channels: list[list[ScoredFact]] = []

        # Channel 1: Semantic search
        query_embedding = self._embed_query(query)
        if query_embedding is not None:
            semantic_results = self.storage.search_vector(query_embedding, limit=25)
        else:
            semantic_results = []
        channels.append(semantic_results)

        # Channel 2: BM25 full-text search
        bm25_results = self.storage.search_bm25(query, limit=10)
        channels.append(bm25_results)

        # Channel 3: Entity graph search
        graph_results = self._entity_graph_search(query, limit=10)
        channels.append(graph_results)

        # Channel 4: Temporal search
        temporal_results = self._temporal_search(query, limit=10)
        channels.append(temporal_results)

        # Step 3: Weighted RRF fusion
        fused = self._weighted_rrf(channels, weights, k=60)

        # Step 4: Format results
        results = fused[:top_k]
        if format == "l0":
            # Return only L0 abstracts (token-efficient mode)
            for r in results:
                if r.fact.l0_abstract:
                    r.fact.content = r.fact.l0_abstract
        return results

    def _classify_intent(self, query: str) -> str:
        """Rule-based intent classification (lightweight, no LLM)."""
        q = query.lower()

        # Temporal signals
        temporal_patterns = [
            r"\bwhen\b", r"\blast time\b", r"\bdate\b", r"\bschedule\b",
            r"\b\d{4}-\d{2}\b", r"\byesterday\b", r"\btomorrow\b",
            r"\blast week\b", r"\bnext week\b", r"\bmonth\b",
        ]
        if any(re.search(p, q) for p in temporal_patterns):
            return IntentType.TEMPORAL

        # Multi-hop signals
        multi_hop_patterns = [
            r"\bwho knows\b.*\bthat\b", r"\bconnect\b.*\bto\b",
            r"\brelat\b.*\bto\b", r"\bthrough\b",
        ]
        if any(re.search(p, q) for p in multi_hop_patterns):
            return IntentType.MULTI_HOP

        # Opinion signals
        opinion_patterns = [
            r"\bthink\b", r"\bfeel\b", r"\bopinion\b", r"\bprefer\b",
            r"\blike\b", r"\bhate\b", r"\bfavorite\b",
        ]
        if any(re.search(p, q) for p in opinion_patterns):
            return IntentType.OPINION

        # Factual signals (check BEFORE entity — "What did X do?" is factual, not entity)
        factual_patterns = [
            r"\bwhat\b", r"\bwho\b", r"\bwhere\b", r"\bhow\b",
            r"\bwhich\b", r"\btell me\b",
        ]
        if any(re.search(p, q) for p in factual_patterns):
            return IntentType.FACTUAL

        # Entity signals (only when no question words present)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        if len(capitalized) >= 2:
            return IntentType.ENTITY

        return IntentType.VAGUE

    def _embed_query(self, query: str) -> list[float] | None:
        """Generate embedding for query."""
        try:
            return self.embedding.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _entity_graph_search(self, query: str, limit: int = 10) -> list[ScoredFact]:
        """Extract entities from query, find matching facts, then traverse graph."""
        # Simple entity extraction: capitalized words
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
        if not entities:
            return []

        all_results: list[ScoredFact] = []
        for entity in entities[:3]:  # Limit entity fan-out
            # Find facts containing this entity
            entity_facts = self.storage.search_by_entity(entity, limit=5)
            all_results.extend(entity_facts)

            # Traverse graph from found facts
            for ef in entity_facts[:2]:
                neighbors = self.storage.get_graph_neighbors(ef.fact.id, hops=2, limit=5)
                all_results.extend(neighbors)

        # Deduplicate by fact ID
        seen = set()
        deduped = []
        for r in all_results:
            if r.fact.id not in seen:
                seen.add(r.fact.id)
                deduped.append(r)
        return deduped[:limit]

    def _temporal_search(self, query: str, limit: int = 10) -> list[ScoredFact]:
        """Extract date references from query and search temporal index."""
        # Simple date extraction
        date_patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{4}-\d{2})\b",
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, query))

        if not dates:
            return []

        # Use first and last date as range
        dates.sort()
        start = dates[0]
        end = dates[-1] if len(dates) > 1 else None
        return self.storage.search_temporal(start, end, limit=limit)

    @staticmethod
    def _weighted_rrf(
        channels: list[list[ScoredFact]],
        weights: list[float],
        k: int = 60,
    ) -> list[ScoredFact]:
        """Weighted Reciprocal Rank Fusion across channels."""
        scores: dict[str, float] = defaultdict(float)
        facts: dict[str, ScoredFact] = {}
        channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        channel_names = ["semantic", "bm25", "graph", "temporal"]

        for channel_idx, (channel_results, weight) in enumerate(zip(channels, weights)):
            if weight == 0:
                continue
            for rank, scored_fact in enumerate(channel_results):
                fid = scored_fact.fact.id
                rrf_score = weight / (k + rank + 1)
                scores[fid] += rrf_score
                channel_scores[fid][channel_names[channel_idx]] = rrf_score
                if fid not in facts:
                    facts[fid] = scored_fact

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for fid, score in ranked:
            sf = facts[fid]
            sf.score = score
            sf.channels = dict(channel_scores[fid])
            results.append(sf)
        return results
