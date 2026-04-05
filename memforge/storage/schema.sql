-- MemForge Schema v1
-- Single-file SQLite with WAL mode, vector search, and FTS5

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- Core: Atomic Facts
CREATE TABLE IF NOT EXISTS atomic_facts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    l0_abstract TEXT,
    fact_type TEXT NOT NULL CHECK(fact_type IN ('episodic','semantic','opinion','temporal','procedural')),
    -- Structured metadata
    entities JSON,
    temporal JSON,
    location TEXT,
    topic TEXT,
    -- Lifecycle
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    trust_score REAL DEFAULT 0.8,
    source_session TEXT,
    -- Scoping
    profile_id TEXT DEFAULT 'default',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 full-text index
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content,
    l0_abstract,
    content='atomic_facts',
    content_rowid='rowid',
    tokenize='unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON atomic_facts BEGIN
    INSERT INTO facts_fts(rowid, content, l0_abstract)
    VALUES (new.rowid, new.content, new.l0_abstract);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON atomic_facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, l0_abstract)
    VALUES('delete', old.rowid, old.content, old.l0_abstract);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON atomic_facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, l0_abstract)
    VALUES('delete', old.rowid, old.content, old.l0_abstract);
    INSERT INTO facts_fts(rowid, content, l0_abstract)
    VALUES (new.rowid, new.content, new.l0_abstract);
END;

-- Graph edges
CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL CHECK(edge_type IN ('entity_shared','temporal_adjacent','semantic_similar','superseded_by')),
    weight REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, edge_type)
);

-- Deduplication hash index
CREATE TABLE IF NOT EXISTS fact_hashes (
    content_hash TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE
);

-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    facts_created INTEGER DEFAULT 0,
    facts_updated INTEGER DEFAULT 0,
    profile_id TEXT DEFAULT 'default'
);

-- Hot path indexes
CREATE INDEX IF NOT EXISTS idx_facts_profile ON atomic_facts(profile_id);
CREATE INDEX IF NOT EXISTS idx_facts_type ON atomic_facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_topic ON atomic_facts(topic);
CREATE INDEX IF NOT EXISTS idx_facts_created ON atomic_facts(created_at);
CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type);
