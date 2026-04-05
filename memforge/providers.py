"""LLM and Embedding provider abstraction.

Supports:
- OpenAI (and any OpenAI-compatible: OpenRouter, DeepSeek, Ollama, vLLM, LM Studio, Together, etc.)
- Anthropic (Claude)
- Google (Gemini)

Configuration via config.toml or environment variables.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Provider configuration — can be loaded from config.toml or constructed directly."""
    provider: str = "openai"          # openai | anthropic | google | openrouter | ollama | custom
    model: str = "gpt-4.1-mini"
    api_key: str | None = None        # Resolved value (prefer api_key_env)
    api_key_env: str | None = None    # Env var name, e.g. "OPENAI_API_KEY"
    base_url: str | None = None       # Override for OpenAI-compatible providers
    extra: dict = field(default_factory=dict)  # Provider-specific options

    def resolve_api_key(self) -> str | None:
        """Resolve API key from env var or direct value."""
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                return key
        if self.api_key:
            return self.api_key
        # Fallback: try common env vars by provider
        fallback_envs = PROVIDER_DEFAULTS.get(self.provider, {}).get("api_key_env")
        if fallback_envs:
            key = os.environ.get(fallback_envs)
            if key:
                return key
        return None

    def resolve_base_url(self) -> str | None:
        """Resolve base URL, using provider defaults if not set."""
        if self.base_url:
            return self.base_url
        return PROVIDER_DEFAULTS.get(self.provider, {}).get("base_url")


# Provider defaults — base_url and default env var for API key
PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,  # OpenAI SDK default
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,  # Uses Anthropic SDK
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": None,  # Uses Google SDK
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
    },
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },
    "custom": {
        "api_key_env": None,
        "base_url": None,
    },
}


# ── LLM Client Abstraction ──────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """Send chat completion request, return response text."""
        ...


class OpenAICompatibleLLM(BaseLLMClient):
    """Works with OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, Together, etc."""

    def __init__(self, config: ProviderConfig):
        from openai import OpenAI
        kwargs = {}
        api_key = config.resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        base_url = config.resolve_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        # Ollama doesn't need an API key, but OpenAI SDK requires one
        if config.provider == "ollama" and "api_key" not in kwargs:
            kwargs["api_key"] = "ollama"
        self.client = OpenAI(**kwargs)
        self.model = config.model

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class AnthropicLLM(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(self, config: ProviderConfig):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install anthropic SDK: pip install anthropic")
        api_key = config.resolve_api_key()
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = config.model

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> str:
        # Anthropic API separates system from messages
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        kwargs = {"model": self.model, "messages": chat_messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class GoogleLLM(BaseLLMClient):
    """Google Gemini client."""

    def __init__(self, config: ProviderConfig):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai SDK: pip install google-genai")
        api_key = config.resolve_api_key()
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = config.model

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> str:
        from google.genai import types
        # Convert OpenAI message format to Gemini
        contents = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            else:
                role = "user" if m["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        return response.text


# ── Embedding Client Abstraction ─────────────────────────────────────────────


class BaseEmbeddingClient(ABC):
    """Abstract embedding client."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


class OpenAICompatibleEmbedding(BaseEmbeddingClient):
    """Works with OpenAI, OpenRouter, Together, etc."""

    def __init__(self, config: ProviderConfig, dimensions: int = 768):
        from openai import OpenAI
        kwargs = {}
        api_key = config.resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        base_url = config.resolve_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        if config.provider == "ollama" and "api_key" not in kwargs:
            kwargs["api_key"] = "ollama"
        self.client = OpenAI(**kwargs)
        self.model = config.model
        self._dim = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = 512
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            kwargs = {"model": self.model, "input": batch}
            # Only pass dimensions if model supports it
            if "text-embedding-3" in self.model or "text-embedding-ada" in self.model:
                kwargs["dimensions"] = self._dim
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


class GoogleEmbedding(BaseEmbeddingClient):
    """Google Gemini embedding client."""

    def __init__(self, config: ProviderConfig, dimensions: int = 768):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai SDK: pip install google-genai")
        api_key = config.resolve_api_key()
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = config.model
        self._dim = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._dim),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


# ── Factory Functions ────────────────────────────────────────────────────────


def create_llm(config: ProviderConfig) -> BaseLLMClient:
    """Create LLM client from config."""
    if config.provider == "anthropic":
        return AnthropicLLM(config)
    elif config.provider == "google":
        return GoogleLLM(config)
    else:
        # OpenAI-compatible: openai, openrouter, deepseek, ollama, together, custom
        return OpenAICompatibleLLM(config)


def create_embedding(config: ProviderConfig, dimensions: int = 768) -> BaseEmbeddingClient:
    """Create embedding client from config."""
    if config.provider == "google":
        return GoogleEmbedding(config, dimensions)
    else:
        return OpenAICompatibleEmbedding(config, dimensions)


# ── Preset Configurations ────────────────────────────────────────────────────


PRESETS: dict[str, dict] = {
    "openai": {
        "llm": ProviderConfig(provider="openai", model="gpt-4.1-mini"),
        "embedding": ProviderConfig(provider="openai", model="text-embedding-3-small"),
    },
    "anthropic": {
        "llm": ProviderConfig(provider="anthropic", model="claude-sonnet-4-6"),
        "embedding": ProviderConfig(provider="openai", model="text-embedding-3-small"),
        # Note: Anthropic doesn't offer embeddings, fall back to OpenAI or use local
    },
    "openrouter": {
        "llm": ProviderConfig(provider="openrouter", model="anthropic/claude-sonnet-4-6"),
        "embedding": ProviderConfig(provider="openrouter", model="openai/text-embedding-3-small"),
    },
    "google": {
        "llm": ProviderConfig(provider="google", model="gemini-2.5-flash"),
        "embedding": ProviderConfig(provider="google", model="gemini-embedding-001"),
    },
    "deepseek": {
        "llm": ProviderConfig(provider="deepseek", model="deepseek-chat"),
        "embedding": ProviderConfig(provider="openai", model="text-embedding-3-small"),
    },
    "ollama": {
        "llm": ProviderConfig(provider="ollama", model="qwen3:1.7b"),
        "embedding": ProviderConfig(provider="ollama", model="nomic-embed-text"),
    },
}


def get_preset(name: str) -> dict[str, ProviderConfig]:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
