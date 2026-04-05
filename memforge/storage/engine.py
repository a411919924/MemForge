"""SQLite storage engine with vector search and FTS5."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from memforge.models import AtomicFact, FactType, ScoredFact, TemporalInfo

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class StorageEngine:
    """Single-file SQLite storage with vector + FTS5 indexes."""

    def __init__(self, db_path: str = "~/.memforge/memforge.db", embedding_dim: int = 1024):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._vec_enabled = False
        self._embedding_dim = embedding_dim

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Try loading sqlite-vec extension
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._vec_enabled = True
        except (ImportError, Exception):
            self._vec_enabled = False
        self._init_schema()

    def _init_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text()
        self._conn.executescript(schema_sql)
        if self._vec_enabled:
            # Create vector table if not exists
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec "
                    f"USING vec0(id TEXT PRIMARY KEY, embedding float[{self._embedding_dim}] distance_metric=cosine)"
                )
                self._conn.commit()
            except Exception:
                # vec table may already exist or vec0 not available
                pass

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    def insert_fact(self, fact: AtomicFact) -> None:
        """Insert an atomic fact with all indexes."""
        d = fact.to_storage_dict()
        self.conn.execute(
            """INSERT OR REPLACE INTO atomic_facts
            (id, content, l0_abstract, fact_type, entities, temporal,
             location, topic, access_count, trust_score, source_session,
             profile_id, created_at, updated_at)
            VALUES (:id, :content, :l0_abstract, :fact_type, :entities, :temporal,
                    :location, :topic, :access_count, :trust_score, :source_session,
                    :profile_id, :created_at, :updated_at)""",
            d,
        )
        # Insert content hash for dedup
        self.conn.execute(
            "INSERT OR IGNORE INTO fact_hashes (content_hash, fact_id) VALUES (?, ?)",
            (fact.content_hash, fact.id),
        )
        # Insert vector embedding
        if self._vec_enabled and fact.embedding is not None:
            vec_bytes = np.array(fact.embedding, dtype=np.float32).tobytes()
            self.conn.execute(
                "INSERT OR REPLACE INTO facts_vec (id, embedding) VALUES (?, ?)",
                (fact.id, vec_bytes),
            )
        self.conn.commit()

    def insert_facts_batch(self, facts: list[AtomicFact]) -> None:
        """Batch insert for efficiency."""
        for fact in facts:
            d = fact.to_storage_dict()
            self.conn.execute(
                """INSERT OR REPLACE INTO atomic_facts
                (id, content, l0_abstract, fact_type, entities, temporal,
                 location, topic, access_count, trust_score, source_session,
                 profile_id, created_at, updated_at)
                VALUES (:id, :content, :l0_abstract, :fact_type, :entities, :temporal,
                        :location, :topic, :access_count, :trust_score, :source_session,
                        :profile_id, :created_at, :updated_at)""",
                d,
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO fact_hashes (content_hash, fact_id) VALUES (?, ?)",
                (fact.content_hash, fact.id),
            )
            if self._vec_enabled and fact.embedding is not None:
                vec_bytes = np.array(fact.embedding, dtype=np.float32).tobytes()
                self.conn.execute(
                    "INSERT OR REPLACE INTO facts_vec (id, embedding) VALUES (?, ?)",
                    (fact.id, vec_bytes),
                )
        self.conn.commit()

    def insert_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO graph_edges (source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            (source_id, target_id, edge_type, weight),
        )
        self.conn.commit()

    def search_vector(self, query_embedding: list[float], limit: int = 25) -> list[ScoredFact]:
        """Semantic search via sqlite-vec cosine distance."""
        if not self._vec_enabled:
            return []
        vec_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        rows = self.conn.execute(
            """SELECT v.id, v.distance, f.*
            FROM facts_vec v
            JOIN atomic_facts f ON v.id = f.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance""",
            (vec_bytes, limit),
        ).fetchall()
        results = []
        for row in rows:
            fact = self._row_to_fact(row)
            # cosine distance → similarity: 1 - distance
            score = 1.0 - row["distance"]
            results.append(ScoredFact(fact=fact, score=score, channels={"semantic": score}))
        return results

    def search_bm25(self, query: str, limit: int = 10) -> list[ScoredFact]:
        """Full-text search via FTS5 BM25."""
        # Sanitize query for FTS5
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        try:
            rows = self.conn.execute(
                """SELECT f.*, bm25(facts_fts) as rank
                FROM facts_fts fts
                JOIN atomic_facts f ON fts.rowid = f.rowid
                WHERE facts_fts MATCH ?
                ORDER BY rank
                LIMIT ?""",
                (sanitized, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results = []
        for row in rows:
            fact = self._row_to_fact(row)
            # BM25 returns negative scores (lower = better), normalize
            bm25_score = -row["rank"]
            score = min(1.0, bm25_score / 10.0)  # rough normalization
            results.append(ScoredFact(fact=fact, score=score, channels={"bm25": score}))
        return results

    def search_by_entity(self, entity: str, limit: int = 10) -> list[ScoredFact]:
        """Find facts mentioning a specific entity."""
        rows = self.conn.execute(
            """SELECT * FROM atomic_facts
            WHERE entities LIKE ?
            ORDER BY created_at DESC
            LIMIT ?""",
            (f'%"{entity}"%', limit),
        ).fetchall()
        return [
            ScoredFact(fact=self._row_to_fact(row), score=0.8, channels={"entity": 0.8})
            for row in rows
        ]

    def get_graph_neighbors(self, fact_id: str, hops: int = 2, limit: int = 10) -> list[ScoredFact]:
        """Graph traversal via spreading activation."""
        visited = set()
        current_layer = {fact_id}
        all_neighbors = []

        for hop in range(hops):
            if not current_layer:
                break
            next_layer = set()
            for node_id in current_layer:
                if node_id in visited:
                    continue
                visited.add(node_id)
                rows = self.conn.execute(
                    """SELECT target_id, weight FROM graph_edges WHERE source_id = ?
                    UNION
                    SELECT source_id, weight FROM graph_edges WHERE target_id = ?""",
                    (node_id, node_id),
                ).fetchall()
                for row in rows:
                    neighbor_id = row[0]
                    if neighbor_id not in visited:
                        weight = row[1]
                        decay = 1.0 / (hop + 1)
                        all_neighbors.append((neighbor_id, weight * decay))
                        next_layer.add(neighbor_id)
            current_layer = next_layer

        # Fetch the actual facts
        all_neighbors.sort(key=lambda x: -x[1])
        results = []
        for neighbor_id, score in all_neighbors[:limit]:
            if neighbor_id == fact_id:
                continue
            row = self.conn.execute(
                "SELECT * FROM atomic_facts WHERE id = ?", (neighbor_id,)
            ).fetchone()
            if row:
                results.append(ScoredFact(
                    fact=self._row_to_fact(row), score=score, channels={"graph": score}
                ))
        return results

    def search_temporal(self, start: str | None, end: str | None, limit: int = 10) -> list[ScoredFact]:
        """Search facts by temporal range."""
        if start and end:
            rows = self.conn.execute(
                """SELECT * FROM atomic_facts
                WHERE json_extract(temporal, '$.referenced_date') BETWEEN ? AND ?
                ORDER BY json_extract(temporal, '$.referenced_date')
                LIMIT ?""",
                (start, end, limit),
            ).fetchall()
        elif start:
            rows = self.conn.execute(
                """SELECT * FROM atomic_facts
                WHERE json_extract(temporal, '$.referenced_date') >= ?
                ORDER BY json_extract(temporal, '$.referenced_date')
                LIMIT ?""",
                (start, limit),
            ).fetchall()
        else:
            return []
        return [
            ScoredFact(fact=self._row_to_fact(row), score=0.7, channels={"temporal": 0.7})
            for row in rows
        ]

    def check_hash_exists(self, content_hash: str) -> str | None:
        """Check if exact content already exists. Returns fact_id or None."""
        row = self.conn.execute(
            "SELECT fact_id FROM fact_hashes WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row["fact_id"] if row else None

    def get_fact(self, fact_id: str) -> AtomicFact | None:
        row = self.conn.execute("SELECT * FROM atomic_facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row else None

    def get_all_facts(self, profile_id: str = "default") -> list[AtomicFact]:
        rows = self.conn.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? ORDER BY created_at",
            (profile_id,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def count_facts(self, profile_id: str = "default") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM atomic_facts WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        return row["cnt"]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_fact(self, row: sqlite3.Row) -> AtomicFact:
        temporal_data = json.loads(row["temporal"]) if row["temporal"] else {}
        entities_data = json.loads(row["entities"]) if row["entities"] else []
        return AtomicFact(
            id=row["id"],
            content=row["content"],
            l0_abstract=row["l0_abstract"] or "",
            fact_type=FactType(row["fact_type"]),
            entities=entities_data,
            temporal=TemporalInfo(**temporal_data),
            location=row["location"],
            topic=row["topic"],
            access_count=row["access_count"],
            trust_score=row["trust_score"],
            source_session=row["source_session"],
            profile_id=row["profile_id"],
        )

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query for FTS5 — use OR logic with prefix matching."""
        import re
        reserved = {"and", "or", "not", "near", "select", "from", "where", "order", "by"}
        stop_words = {"a", "an", "the", "is", "was", "were", "are", "be", "been",
                      "do", "did", "does", "has", "had", "have", "will", "would",
                      "could", "should", "may", "might", "what", "when", "who",
                      "how", "which", "that", "this", "to", "of", "in", "for",
                      "on", "at", "it", "its", "i", "you", "he", "she", "we", "they"}
        words = query.split()
        cleaned = []
        for w in words:
            w = re.sub(r"[^\w]", "", w)  # strip punctuation
            if w and w.lower() not in reserved and w.lower() not in stop_words:
                cleaned.append(f'"{w}"*')  # prefix match with quoting
        if not cleaned:
            return ""
        return " OR ".join(cleaned)
