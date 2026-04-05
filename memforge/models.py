"""Core data models for MemForge."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime
from enum import Enum
from typing import Any


class FactType(str, Enum):
    EPISODIC = "episodic"        # Events: "Alice met Bob on 2026-04-05"
    SEMANTIC = "semantic"        # Knowledge: "Python is a programming language"
    OPINION = "opinion"          # Preferences: "Alice prefers dark mode"
    TEMPORAL = "temporal"        # Time-bound: "The meeting is scheduled for 2026-04-10"
    PROCEDURAL = "procedural"    # How-to: "To deploy, run `make deploy`"


class ConflictAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    NOOP = "noop"


@dataclass
class TemporalInfo:
    """Three-date temporal model following SLM V3."""
    observation_date: str | None = None      # When the fact was observed
    referenced_date: str | None = None       # What date the fact refers to
    interval_start: str | None = None        # Start of a time range
    interval_end: str | None = None          # End of a time range

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AtomicFact:
    """A single, self-contained, atomic memory unit."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                        # Full self-contained fact
    l0_abstract: str = ""                    # ~20 token summary
    fact_type: FactType = FactType.SEMANTIC

    # Structured metadata
    entities: list[str] = field(default_factory=list)
    temporal: TemporalInfo = field(default_factory=TemporalInfo)
    location: str | None = None
    topic: str | None = None

    # Embedding (set after creation)
    embedding: list[float] | None = None

    # Lifecycle
    access_count: int = 0
    trust_score: float = 0.8
    source_session: str | None = None

    # Scoping
    profile_id: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def content_hash(self) -> str:
        """SHA256 hash of normalized content for exact dedup."""
        normalized = self.content.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dict for SQLite insertion."""
        import json
        return {
            "id": self.id,
            "content": self.content,
            "l0_abstract": self.l0_abstract,
            "fact_type": self.fact_type.value,
            "entities": json.dumps(self.entities),
            "temporal": json.dumps(self.temporal.to_dict()),
            "location": self.location,
            "topic": self.topic,
            "access_count": self.access_count,
            "trust_score": self.trust_score,
            "source_session": self.source_session,
            "profile_id": self.profile_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ScoredFact:
    """A fact with retrieval score and channel attribution."""
    fact: AtomicFact
    score: float
    channels: dict[str, float] = field(default_factory=dict)  # channel_name → score


@dataclass
class Message:
    """A single conversation message."""
    role: str        # "user" | "assistant" | "system"
    content: str
    timestamp: datetime | None = None


@dataclass
class ConflictResult:
    """Result of conflict detection for a candidate fact."""
    action: ConflictAction
    candidate: AtomicFact
    existing_fact: AtomicFact | None = None   # The fact it conflicts/duplicates with
    similarity: float = 0.0
