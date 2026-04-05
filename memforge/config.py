"""YAML-based configuration for MemForge.

Config file location (in priority order):
1. Explicit path passed to MemForge()
2. ./memforge.yaml (project-local)
3. ~/.memforge/config.yaml (global)

Example config:
    llm:
      provider: openrouter
      model: anthropic/claude-sonnet-4-6
      api_key_env: OPENROUTER_API_KEY

    embedding:
      provider: openrouter
      model: openai/text-embedding-3-small
      api_key_env: OPENROUTER_API_KEY
      dimensions: 768

    storage:
      path: ~/.memforge/memforge.db

    ingestion:
      window_size: 40
      window_overlap: 2
      dedup_threshold: 0.85

    retrieval:
      top_k: 10
      channels: [semantic, bm25, graph, temporal]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memforge.providers import ProviderConfig

# Search order for config files
CONFIG_SEARCH_PATHS = [
    Path("./memforge.yaml"),
    Path("./memforge.yml"),
    Path("~/.memforge/config.yaml").expanduser(),
    Path("~/.memforge/config.yml").expanduser(),
]


@dataclass
class IngestionConfig:
    window_size: int = 40
    window_overlap: int = 2
    dedup_threshold: float = 0.85


@dataclass
class RetrievalConfig:
    top_k: int = 10
    channels: list[str] = field(default_factory=lambda: ["semantic", "bm25", "graph", "temporal"])


@dataclass
class MemForgeConfig:
    """Full MemForge configuration."""
    ingestion_llm: ProviderConfig = field(default_factory=lambda: ProviderConfig())
    qa_llm: ProviderConfig | None = None       # Falls back to ingestion_llm
    judge_llm: ProviderConfig | None = None     # Falls back to qa_llm
    embedding: ProviderConfig = field(default_factory=lambda: ProviderConfig())
    embedding_dim: int = 768
    storage_path: str = "~/.memforge/memforge.db"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Legacy compat: single "llm" maps to ingestion_llm
    @property
    def llm(self) -> ProviderConfig:
        return self.ingestion_llm


def _dict_to_provider_config(d: dict[str, Any]) -> ProviderConfig:
    """Convert a dict from YAML to ProviderConfig."""
    return ProviderConfig(
        provider=d.get("provider", "openai"),
        model=d.get("model", "gpt-4.1-mini"),
        api_key=d.get("api_key"),
        api_key_env=d.get("api_key_env"),
        base_url=d.get("base_url"),
        extra=d.get("extra", {}),
    )


def load_config(path: str | Path | None = None) -> MemForgeConfig:
    """Load config from YAML file.

    Args:
        path: Explicit path to config file. If None, searches default locations.

    Returns:
        MemForgeConfig with values from file (or defaults if no file found).
    """
    import yaml

    config_path = _find_config(path)
    if config_path is None:
        return MemForgeConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return _parse_raw_config(raw)


def _find_config(path: str | Path | None) -> Path | None:
    """Find config file from explicit path or search defaults."""
    if path is not None:
        p = Path(path).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    for candidate in CONFIG_SEARCH_PATHS:
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved
    return None


def _parse_raw_config(raw: dict[str, Any]) -> MemForgeConfig:
    """Parse raw YAML dict into MemForgeConfig."""
    config = MemForgeConfig()

    # LLM configs (support both new multi-role and legacy single "llm")
    if "ingestion_llm" in raw:
        config.ingestion_llm = _dict_to_provider_config(raw["ingestion_llm"])
    elif "llm" in raw:
        config.ingestion_llm = _dict_to_provider_config(raw["llm"])

    if "qa_llm" in raw:
        config.qa_llm = _dict_to_provider_config(raw["qa_llm"])

    if "judge_llm" in raw:
        config.judge_llm = _dict_to_provider_config(raw["judge_llm"])

    # Embedding config
    if "embedding" in raw:
        emb = raw["embedding"]
        config.embedding = _dict_to_provider_config(emb)
        if "dimensions" in emb:
            config.embedding_dim = emb["dimensions"]

    # Storage
    if "storage" in raw:
        config.storage_path = raw["storage"].get("path", config.storage_path)

    # Ingestion
    if "ingestion" in raw:
        ing = raw["ingestion"]
        config.ingestion = IngestionConfig(
            window_size=ing.get("window_size", 40),
            window_overlap=ing.get("window_overlap", 2),
            dedup_threshold=ing.get("dedup_threshold", 0.85),
        )

    # Retrieval
    if "retrieval" in raw:
        ret = raw["retrieval"]
        config.retrieval = RetrievalConfig(
            top_k=ret.get("top_k", 10),
            channels=ret.get("channels", ["semantic", "bm25", "graph", "temporal"]),
        )

    return config
