# RFC-001: MemForge Architecture Design

> Status: Draft
> Author: @xmfy
> Created: 2026-04-05
> Target: LoCoMo SOTA (> 86.23%) with flexible integration (Claude Code / OpenClaw / any MCP client)

---

## 1. Problem Statement

Current agent memory systems suffer from a fundamental imbalance: **retrieval is over-engineered while ingestion is under-engineered**. The best retrieval system cannot compensate for low-quality memory writes.

| System | Ingestion Quality | Retrieval Quality | Integration | LoCoMo |
|--------|-------------------|-------------------|-------------|--------|
| Mem0 | Low (raw LLM summary) | Medium (vector only) | High (MCP + plugins) | ~64% |
| OMEGA | Low (regex capture) | High (vector + FTS + rerank) | Highest (MCP + 11 hooks) | — |
| SimpleMem | **Highest** (atomic facts + coref + temporal) | Medium (3-view parallel) | None | 43.24% (token-efficient) |
| ReMe | Medium (categorized summaries) | Medium (vector + BM25) | Medium (AgentScope) | 86.23% |
| SLM V3 | High (11-step pipeline) | **Highest** (6-channel) | None | 74.8% (zero-LLM) |

**MemForge thesis**: Combine SimpleMem-grade ingestion with SLM V3-grade retrieval and OMEGA-grade integration in a single, cohesive system.

---

## 2. Design Principles

1. **Ingestion-first** — Memory quality is determined at write time, not read time
2. **Single-file storage** — SQLite-WAL, zero external services, `~/.memforge/memforge.db`
3. **Graceful degradation** — Cloud LLM → Small local LLM → Zero-LLM, same interface
4. **Framework-agnostic** — MCP Server as primary interface, Hooks as optional accelerator
5. **Benchmark-driven** — Every design decision must be measurable on LoCoMo

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Integration Layer                         │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ MCP Server  │  │ Hooks Engine │  │ File Bridge         │ │
│  │ (stdio/SSE) │  │ (Claude Code)│  │ (CLAUDE.md ↔ sync)  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
├─────────┼────────────────┼──────────────────────┼────────────┤
│         └────────────────┼──────────────────────┘            │
│                          ▼                                    │
│                   ┌─────────────┐                             │
│                   │  MemForge   │                             │
│                   │    Core     │                             │
│                   └──────┬──────┘                             │
│                          │                                    │
│              ┌───────────┼───────────┐                        │
│              ▼           ▼           ▼                        │
│  ┌───────────────┐ ┌──────────┐ ┌───────────────┐           │
│  │   Ingestion   │ │ Storage  │ │   Retrieval   │           │
│  │   Pipeline    │ │  Engine  │ │    Engine     │           │
│  │               │ │          │ │               │           │
│  │ FactExtractor │ │ SQLite   │ │ Semantic      │           │
│  │ CorefResolver │ │ WAL +    │ │ BM25/FTS5     │           │
│  │ TemporalNorm  │ │ vec +    │ │ EntityGraph   │           │
│  │ ConflictDetect│ │ FTS5 +   │ │ Temporal      │           │
│  │ LayerGen      │ │ Edges    │ │ CrossEncoder  │           │
│  └───────────────┘ └──────────┘ └───────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Ingestion Pipeline (The Core Innovation)

### 4.1 Overview

```
Raw Messages
    │
    ▼
┌─────────────────────┐
│ Sliding Window       │  40 messages/window, 2-message overlap
│ Segmentation         │  (SimpleMem strategy)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Fact Extraction      │  Mode A: regex + templates (zero-LLM)
│ (Atomization)        │  Mode B: small LLM (Qwen3-1.7B)
│                      │  Mode C: cloud LLM (GPT-4.1-mini/Claude)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Coreference          │  "he" → "Alice", "it" → "the database"
│ Resolution           │  Uses conversation context + entity tracking
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Temporal             │  "tomorrow" → "2026-04-06"
│ Normalization        │  "last week" → "2026-03-29"
│                      │  3-date model: observation, referenced, interval
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Conflict Detection   │  Vector search existing facts
│ & Deduplication      │  Decision: ADD / UPDATE / SUPERSEDE / NOOP
│                      │  SHA256 exact match → semantic similarity → Jaccard
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ L0 Summary           │  Generate ~20 token abstract per fact
│ Generation           │  For token-efficient retrieval
└─────────┬───────────┘
          ▼
    Atomic Facts → Storage
```

### 4.2 Fact Extraction Modes

#### Mode C (Cloud LLM) — Highest Quality

```python
EXTRACTION_PROMPT = """
Given the following conversation segment, extract ALL atomic facts.

Rules:
1. Each fact must be self-contained (no pronouns, no relative time)
2. Resolve all coreferences: replace pronouns with actual names/entities
3. Normalize all temporal expressions to ISO 8601 absolute dates
4. Classify each fact: episodic | semantic | opinion | temporal | procedural
5. Include structured metadata: entities, timestamps, locations, topics

Conversation:
{window}

Previous facts (for dedup reference):
{prev_facts}

Output JSON array of facts:
[{
  "content": "self-contained atomic fact",
  "fact_type": "episodic|semantic|opinion|temporal|procedural",
  "entities": ["entity1", "entity2"],
  "temporal": {"referenced_date": "ISO8601 or null", "interval_start": null, "interval_end": null},
  "location": "location or null",
  "topic": "topic"
}]
"""
```

#### Mode A (Zero-LLM) — Offline Fallback

```python
class RuleBasedExtractor:
    """
    Fact extraction without any LLM:
    1. Sentence splitting (spaCy)
    2. NER for entities (spaCy en_core_web_sm)
    3. Regex temporal parsing (dateutil + custom patterns)
    4. Keyword-based fact type classification
    5. Template-based fact formatting
    """
```

#### Mode B (Small LLM) — The Sweet Spot (Future P3)

Fine-tuned Qwen3-1.7B on Mode C outputs (teacher-student distillation).
Target: 80%+ of Mode C quality at zero cloud cost.

### 4.3 Why This Wins on LoCoMo

LoCoMo evaluates on 5 question types:
1. **Single-hop factual** — Atomic fact extraction directly answers these
2. **Multi-hop reasoning** — Entity graph traversal connects facts
3. **Temporal reasoning** — 3-date temporal model enables time queries
4. **Open-ended** — L0 summaries provide concise, relevant context
5. **Adversarial (unanswerable)** — Conflict detection reduces hallucination

SimpleMem achieves 43.24% F1 with ~550 tokens primarily through superior ingestion.
ReMe achieves 86.23% with full context (but high token cost).
**MemForge targets 88-90%** by combining high-quality ingestion with multi-channel retrieval.

---

## 5. Storage Engine

### 5.1 Schema

```sql
-- Core: Atomic Facts
CREATE TABLE atomic_facts (
    id TEXT PRIMARY KEY,             -- UUID v7 (time-ordered)
    content TEXT NOT NULL,           -- Self-contained atomic fact
    l0_abstract TEXT,                -- ~20 token summary
    fact_type TEXT NOT NULL,         -- episodic/semantic/opinion/temporal/procedural
    -- Embedding
    embedding BLOB,                  -- 768d float32 vector
    -- Structured metadata (indexed)
    entities JSON,                   -- ["Alice", "Starbucks"]
    temporal JSON,                   -- {"referenced_date": "2026-04-06", ...}
    location TEXT,
    topic TEXT,
    -- Lifecycle
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    trust_score REAL DEFAULT 0.8,
    source_session TEXT,             -- session that created this fact
    -- Scoping
    profile_id TEXT DEFAULT 'default',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 full-text index
CREATE VIRTUAL TABLE facts_fts USING fts5(
    content,
    l0_abstract,
    tokenize='unicode61'
);

-- Vector index (sqlite-vec)
CREATE VIRTUAL TABLE facts_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

-- Graph edges for entity/temporal/semantic relationships
CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,         -- entity_shared/temporal_adjacent/semantic_similar/superseded_by
    weight REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, edge_type)
);

-- Deduplication hash index
CREATE TABLE fact_hashes (
    content_hash TEXT PRIMARY KEY,   -- SHA256 of normalized content
    fact_id TEXT NOT NULL REFERENCES atomic_facts(id) ON DELETE CASCADE
);

-- Session tracking
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    facts_created INTEGER DEFAULT 0,
    facts_updated INTEGER DEFAULT 0,
    profile_id TEXT DEFAULT 'default'
);

-- Indexes for hot paths
CREATE INDEX idx_facts_profile ON atomic_facts(profile_id);
CREATE INDEX idx_facts_type ON atomic_facts(fact_type);
CREATE INDEX idx_facts_topic ON atomic_facts(topic);
CREATE INDEX idx_facts_created ON atomic_facts(created_at);
CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
CREATE INDEX idx_edges_type ON graph_edges(edge_type);
```

### 5.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | SQLite-WAL | Zero ops, single file, concurrent reads. OMEGA validated this at scale |
| Vector index | sqlite-vec | No external service. Sufficient for <1M facts |
| FTS | FTS5 unicode61 | Built-in, fast BM25. Chinese tokenizer deferred to P4 |
| Graph | Adjacency table | Lightweight, no Neo4j dependency. 2-hop traversal sufficient |
| IDs | UUID v7 | Time-ordered, globally unique, sortable |
| Embedding dim | 768 | nomic-embed-text-v1.5 (local ONNX) or API-based |

---

## 6. Retrieval Engine

### 6.1 Four-Channel Hybrid Retrieval

```python
class RetrievalEngine:
    async def search(self, query: str, top_k: int = 10) -> list[ScoredFact]:
        # Step 1: Intent classification (lightweight, rule-based)
        intent = self.classify_intent(query)

        # Step 2: Parallel 4-channel retrieval
        semantic, bm25, graph, temporal = await asyncio.gather(
            self.semantic_search(query, limit=25),
            self.bm25_search(query, limit=10),
            self.entity_graph_search(query, hops=2, limit=10),
            self.temporal_search(query, limit=10),
        )

        # Step 3: Intent-adaptive Weighted RRF fusion
        weights = INTENT_WEIGHTS[intent]
        fused = self.weighted_rrf(
            channels=[semantic, bm25, graph, temporal],
            weights=weights,
            k=60
        )

        # Step 4: Cross-encoder reranking (optional, ONNX)
        if self.reranker_available:
            fused = self.cross_encoder_rerank(query, fused[:30])

        # Step 5: Return with L0 abstracts (token-efficient mode)
        return fused[:top_k]
```

### 6.2 Intent-Adaptive Weights

| Query Intent | Semantic | BM25 | Graph | Temporal | Detection Pattern |
|-------------|----------|------|-------|----------|-------------------|
| factual | 0.5 | 0.3 | 0.1 | 0.1 | "what is", "who is", "where" |
| temporal | 0.2 | 0.2 | 0.1 | 0.5 | "when", "last time", date mentions |
| entity | 0.3 | 0.1 | 0.5 | 0.1 | named entities detected |
| multi-hop | 0.3 | 0.1 | 0.4 | 0.2 | "who knows someone that", compound |
| opinion | 0.5 | 0.3 | 0.2 | 0.0 | "think", "feel", "opinion" |
| vague | 0.4 | 0.2 | 0.2 | 0.2 | fallback |

### 6.3 Weighted Reciprocal Rank Fusion

```python
def weighted_rrf(channels: list[list[ScoredFact]],
                 weights: list[float],
                 k: int = 60) -> list[ScoredFact]:
    """
    Fuse multiple ranked lists with channel-specific weights.
    Score = sum(w_i / (k + rank_i)) for each channel where fact appears.
    """
    scores = defaultdict(float)
    facts = {}
    for channel, weight in zip(channels, weights):
        for rank, fact in enumerate(channel):
            scores[fact.id] += weight / (k + rank + 1)
            facts[fact.id] = fact
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [facts[fid] for fid, _ in ranked]
```

---

## 7. Integration Layer

### 7.1 MCP Server (Primary Interface)

```json
{
  "tools": [
    {
      "name": "memforge_store",
      "description": "Store memories from conversation messages",
      "inputSchema": {
        "messages": "array of {role, content} messages",
        "session_id": "optional session identifier",
        "mode": "optional: auto|cloud|local|zero (default: auto)"
      }
    },
    {
      "name": "memforge_query",
      "description": "Search memories by natural language query",
      "inputSchema": {
        "query": "natural language search query",
        "top_k": "number of results (default: 10)",
        "format": "full|l0|structured (default: l0)"
      }
    },
    {
      "name": "memforge_timeline",
      "description": "Browse memories chronologically",
      "inputSchema": {
        "start": "ISO 8601 start date",
        "end": "ISO 8601 end date",
        "topic": "optional topic filter"
      }
    },
    {
      "name": "memforge_debug",
      "description": "Explain why specific facts were retrieved or missed",
      "inputSchema": {
        "query": "the search query",
        "fact_id": "optional: specific fact to explain"
      }
    },
    {
      "name": "memforge_stats",
      "description": "Memory statistics and health metrics",
      "inputSchema": {}
    }
  ]
}
```

### 7.2 Claude Code Hooks (Optional Accelerator)

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "memforge hook session-start --session $SESSION_ID",
        "description": "Load relevant memories for session context"
      }
    ],
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "memforge hook capture --input \"$USER_PROMPT\"",
        "description": "Auto-capture decisions and lessons from user messages"
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "memforge hook post-tool --tool $TOOL_NAME --output \"$TOOL_OUTPUT\"",
        "description": "Surface relevant memories after tool execution"
      }
    ],
    "Stop": [
      {
        "type": "command",
        "command": "memforge hook session-end --session $SESSION_ID",
        "description": "Compress and persist session memories"
      }
    ]
  }
}
```

### 7.3 File Bridge (Claude Code Native Sync)

```
~/.claude/CLAUDE.md          ←── Top-priority facts auto-synced here
~/.memforge/
├── memforge.db              ←── SQLite database
├── config.toml              ←── Configuration
└── exports/
    └── MEMORY.md            ←── Human-readable memory index
```

The File Bridge watches for changes in both directions:
- DB → File: High-trust facts are exported to CLAUDE.md for native loading
- File → DB: Manual edits to MEMORY.md are ingested back into the database

---

## 8. Configuration

```toml
# ~/.memforge/config.toml

[ingestion]
mode = "auto"                        # auto | cloud | local | zero
sliding_window_size = 40
sliding_window_overlap = 2

[ingestion.cloud]
provider = "openai"                  # openai | anthropic
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"       # env var name, not the key itself

[ingestion.local]
model = "qwen3-1.7b"                # future: fine-tuned fact extractor
endpoint = "http://localhost:11434"   # ollama / vllm / etc.

[embedding]
provider = "local"                   # local | openai | custom
model = "nomic-embed-text-v1.5"     # 768d, ONNX runtime
# model = "text-embedding-3-small"  # if provider = openai

[retrieval]
channels = ["semantic", "bm25", "graph", "temporal"]
reranker = "cross-encoder"           # cross-encoder | none
reranker_model = "ms-marco-MiniLM-L-12-v2"
top_k = 10

[storage]
path = "~/.memforge/memforge.db"
profile = "default"

[integration]
mcp_transport = "stdio"              # stdio | sse | http
hooks_enabled = true
file_bridge_enabled = true
claude_md_sync = true
```

---

## 9. LoCoMo Evaluation Plan

### 9.1 Baseline Experiment (P0)

**Goal**: Validate architecture with Mode C (cloud LLM), measure LoCoMo score.

```
Step 1: Implement ingestion pipeline (Mode C only)
Step 2: Implement SQLite storage with all indexes
Step 3: Implement 4-channel retrieval
Step 4: Run LoCoMo evaluation
Step 5: Compare against published baselines
```

**Minimum Viable Pipeline for Baseline**:
- Ingestion: Mode C with GPT-4.1-mini
- Embedding: text-embedding-3-small (API, simplest setup)
- Retrieval: semantic + BM25 (skip graph/temporal for baseline)
- No reranker (add later for ablation)

### 9.2 Ablation Plan

| Experiment | What changes | Expected impact |
|------------|-------------|-----------------|
| Base | Semantic only, raw storage | Baseline |
| +Atomization | Add fact extraction pipeline | +5-10pp (SimpleMem's key insight) |
| +BM25 | Add FTS5 channel | +2-3pp |
| +Graph | Add entity graph traversal | +2-3pp (multi-hop questions) |
| +Temporal | Add temporal channel | +1-2pp (temporal questions) |
| +Reranker | Add cross-encoder | +2-3pp |
| +L0 | Token-efficient mode | Same F1, 30x fewer tokens |

### 9.3 Target Metrics

| Metric | Target | Current SOTA |
|--------|--------|-------------|
| LoCoMo F1 (full) | > 88% | ReMe 86.23% |
| LoCoMo F1 (token-efficient) | > 45% @ <500 tokens | SimpleMem 43.24% @ 550 tokens |
| Ingestion latency (per session) | < 30s | — |
| Query latency (p95) | < 200ms | — |
| Storage size (10K facts) | < 50MB | — |

---

## 10. Project Structure

```
memforge/
├── docs/
│   └── RFC-001-architecture.md     # This document
├── memforge/
│   ├── __init__.py
│   ├── core.py                     # MemForge main class
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Orchestrator
│   │   ├── fact_extractor.py       # Mode A/B/C extractors
│   │   ├── coref_resolver.py       # Coreference resolution
│   │   ├── temporal_normalizer.py  # Temporal normalization
│   │   ├── conflict_detector.py    # Dedup + conflict detection
│   │   └── layer_generator.py      # L0 summary generation
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── engine.py               # SQLite storage engine
│   │   ├── schema.sql              # DDL
│   │   └── migrations/             # Schema migrations
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── engine.py               # Multi-channel orchestrator
│   │   ├── semantic.py             # Vector search
│   │   ├── bm25.py                 # FTS5 search
│   │   ├── graph.py                # Entity graph traversal
│   │   ├── temporal.py             # Temporal search
│   │   ├── fusion.py               # Weighted RRF
│   │   └── reranker.py             # Cross-encoder reranking
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── mcp_server.py           # MCP protocol server
│   │   ├── hooks.py                # Claude Code hooks
│   │   └── file_bridge.py          # CLAUDE.md sync
│   └── models.py                   # Data models (AtomicFact, etc.)
├── eval/
│   ├── locomo/
│   │   ├── run_eval.py             # LoCoMo evaluation runner
│   │   ├── metrics.py              # F1, precision, recall
│   │   └── baselines.py            # Compare against published results
│   └── ablation.py                 # Ablation study runner
├── tests/
├── pyproject.toml
└── README.md
```

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM extraction cost on LoCoMo (10 conversations × N turns) | Budget overrun | Use GPT-4.1-mini (cheap), cache extraction results |
| sqlite-vec performance at scale | Slow retrieval | Benchmark at 10K/100K facts; fall back to FAISS if needed |
| LoCoMo evaluation methodology differences | Unfair comparison | Use exact same eval script as ReMe/SimpleMem |
| Fact atomization may fragment context | Lower F1 on open-ended questions | L0 summaries preserve high-level context |
| Graph construction quality | Poor multi-hop performance | Start with entity co-occurrence (simple), upgrade later |

---

## 12. Non-Goals (Explicitly Out of Scope)

- Chinese language support (P4)
- Multi-user / multi-tenant (not needed for benchmark)
- Cloud deployment / horizontal scaling
- GUI / web dashboard
- Fine-tuned small model (P3)
- Audio/video/image memory
