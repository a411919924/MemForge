"""Fact extraction from conversation messages — the core innovation of MemForge.

Supports three modes:
- Mode C (cloud): GPT-4.1-mini / Claude for highest quality extraction
- Mode B (local): Small local LLM (Qwen3-1.7B) — future, not implemented yet
- Mode A (zero):  Rule-based extraction — future, not implemented yet
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from openai import OpenAI

from memforge.models import AtomicFact, FactType, Message, TemporalInfo

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a memory extraction engine. Given a conversation segment, extract ALL atomic facts.

## Rules
1. Each fact MUST be completely self-contained — no pronouns, no relative time, no ambiguity.
2. Resolve ALL coreferences: replace "he/she/it/they" with actual names/entities.
3. Normalize ALL temporal expressions to ISO 8601 absolute dates. Use the observation date as anchor.
4. Classify each fact into exactly one type:
   - episodic: specific events ("Alice met Bob at Starbucks on 2026-04-05")
   - semantic: general knowledge ("Python uses indentation for blocks")
   - opinion: preferences/feelings ("Alice prefers dark mode")
   - temporal: time-bound scheduled items ("The team meeting is at 2026-04-10T14:00:00")
   - procedural: how-to knowledge ("To deploy, run `make deploy` from project root")
5. Extract entities, temporal info, location, and topic for each fact.
6. Do NOT extract trivial conversational filler ("okay", "thanks", "got it").
7. If a previous fact is updated or contradicted, extract the NEW version only.

## Observation Date
{observation_date}

## Previous Facts (for dedup — do not re-extract these)
{prev_facts}

## Conversation Segment
{conversation}

## Output Format
Return a JSON array. Each element:
{{
  "content": "self-contained atomic fact with all coreferences resolved and dates absolute",
  "fact_type": "episodic|semantic|opinion|temporal|procedural",
  "entities": ["entity1", "entity2"],
  "temporal": {{
    "observation_date": "ISO8601 or null",
    "referenced_date": "ISO8601 or null",
    "interval_start": "ISO8601 or null",
    "interval_end": "ISO8601 or null"
  }},
  "location": "location or null",
  "topic": "brief topic label"
}}

Output ONLY the JSON array, no other text."""

L0_PROMPT = """\
Generate a single-sentence summary (under 20 words) of this fact:

Fact: {content}

Summary:"""


class CloudFactExtractor:
    """Mode C: Cloud LLM-based fact extraction (highest quality)."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def extract(
        self,
        messages: list[Message],
        prev_facts: list[AtomicFact] | None = None,
        observation_date: str | None = None,
    ) -> list[AtomicFact]:
        """Extract atomic facts from a conversation window."""
        if not messages:
            return []

        if observation_date is None:
            observation_date = datetime.utcnow().strftime("%Y-%m-%d")

        # Format conversation
        conv_text = "\n".join(
            f"[{m.role}]: {m.content}" for m in messages
        )

        # Format previous facts for dedup
        prev_text = "None"
        if prev_facts:
            prev_text = "\n".join(f"- {f.content}" for f in prev_facts[-20:])

        prompt = EXTRACTION_PROMPT.format(
            observation_date=observation_date,
            prev_facts=prev_text,
            conversation=conv_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"} if "gpt" in self.model else None,
            )
            content = response.choices[0].message.content.strip()
            facts_data = self._parse_response(content)
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

        # Convert to AtomicFact objects
        facts = []
        for fd in facts_data:
            try:
                temporal = TemporalInfo(
                    observation_date=fd.get("temporal", {}).get("observation_date"),
                    referenced_date=fd.get("temporal", {}).get("referenced_date"),
                    interval_start=fd.get("temporal", {}).get("interval_start"),
                    interval_end=fd.get("temporal", {}).get("interval_end"),
                )
                fact = AtomicFact(
                    content=fd["content"],
                    fact_type=FactType(fd.get("fact_type", "semantic")),
                    entities=fd.get("entities", []),
                    temporal=temporal,
                    location=fd.get("location"),
                    topic=fd.get("topic"),
                )
                facts.append(fact)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed fact: {e}")
                continue

        return facts

    def generate_l0(self, facts: list[AtomicFact]) -> list[AtomicFact]:
        """Generate L0 abstracts for a batch of facts."""
        for fact in facts:
            if fact.l0_abstract:
                continue
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": L0_PROMPT.format(content=fact.content)}],
                    temperature=0.0,
                    max_tokens=50,
                )
                fact.l0_abstract = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"L0 generation failed for fact {fact.id}: {e}")
                # Fallback: truncate content
                fact.l0_abstract = fact.content[:80] + "..." if len(fact.content) > 80 else fact.content
        return facts

    @staticmethod
    def _parse_response(content: str) -> list[dict]:
        """Parse LLM JSON response, handling common quirks."""
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]

        parsed = json.loads(content)

        # Handle both {"facts": [...]} and [...] formats
        if isinstance(parsed, dict):
            for key in ("facts", "results", "data", "memories"):
                if key in parsed:
                    return parsed[key]
            return []
        elif isinstance(parsed, list):
            return parsed
        return []
