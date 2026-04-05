"""Fact extraction from conversation messages — the core innovation of MemForge.

Supports three modes:
- Mode C (cloud): GPT-4.1-mini / Claude / Gemini / OpenRouter for highest quality extraction
- Mode B (local): Small local LLM (Qwen3-1.7B) — future, not implemented yet
- Mode A (zero):  Rule-based extraction — future, not implemented yet
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from memforge.models import AtomicFact, FactType, Message, TemporalInfo
from memforge.providers import BaseLLMClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a memory extraction engine. Given a conversation segment, extract ALL atomic facts — both explicit and implied.

## Rules
1. Each fact MUST be completely self-contained — no pronouns, no relative time, no ambiguity.
2. Resolve ALL coreferences: replace "he/she/it/they" with actual names/entities.
3. Normalize ALL temporal expressions to ISO 8601 absolute dates. Use the observation date as anchor.
4. Classify each fact into exactly one type:
   - episodic: specific events ("Alice met Bob at Starbucks on 2026-04-05")
   - semantic: general knowledge / personal attributes ("Alice's home country is Sweden")
   - opinion: preferences/feelings ("Alice prefers dark mode")
   - temporal: time-bound scheduled items ("The team meeting is at 2026-04-10T14:00:00")
   - procedural: how-to knowledge ("To deploy, run `make deploy` from project root")
5. Extract entities, temporal info, location, and topic for each fact.
6. Do NOT extract trivial conversational filler ("okay", "thanks", "got it").
7. If a previous fact is updated or contradicted, extract the NEW version only.
8. CRITICAL — Extract IMPLIED facts by combining context clues:
   - If someone mentions "my grandmother in Sweden" and "I moved from my home country", infer "X's home country is Sweden"
   - If someone says "as a single parent", infer "X is single" and "X is a parent"
   - If someone discusses "my transition" in LGBTQ context, infer "X is transgender"
   - Extract personal attributes: nationality, relationship status, occupation, hobbies, age, identity
   - Extract each person's activities, hobbies, interests as separate facts
9. Be EXHAUSTIVE — extract MORE facts rather than fewer. It's better to have redundant facts than to miss important ones.

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


class FactExtractor:
    """LLM-based fact extraction — works with any provider via BaseLLMClient."""

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

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
            content = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=32768,
            )
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

    def generate_l0(self, facts: list[AtomicFact], max_workers: int = 8) -> list[AtomicFact]:
        """Generate L0 abstracts for a batch of facts (parallelized)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        pending = [f for f in facts if not f.l0_abstract]
        if not pending:
            return facts

        def _gen_one(fact: AtomicFact) -> None:
            try:
                fact.l0_abstract = self.llm.chat(
                    messages=[{"role": "user", "content": L0_PROMPT.format(content=fact.content)}],
                    temperature=0.0,
                    max_tokens=50,
                )
            except Exception as e:
                logger.warning(f"L0 generation failed for fact {fact.id}: {e}")
                fact.l0_abstract = fact.content[:80] + ("..." if len(fact.content) > 80 else "")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_gen_one, f) for f in pending]
            with tqdm(total=len(pending), desc="Generating L0", unit="fact") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        return facts

    @staticmethod
    def _parse_response(content: str) -> list[dict]:
        """Parse LLM JSON response, handling truncation and common quirks."""
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
        content = content.strip()

        # Try direct parse first
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Truncated JSON — try to salvage by finding the last complete object
            parsed = FactExtractor._repair_truncated_json(content)

        # Handle both {"facts": [...]} and [...] formats
        if isinstance(parsed, dict):
            for key in ("facts", "results", "data", "memories"):
                if key in parsed:
                    return parsed[key]
            return []
        elif isinstance(parsed, list):
            return parsed
        return []

    @staticmethod
    def _repair_truncated_json(content: str) -> list:
        """Attempt to repair truncated JSON array by finding last complete object."""
        # Find the last complete "}" and close the array
        last_brace = content.rfind("}")
        if last_brace == -1:
            return []
        truncated = content[:last_brace + 1]
        # Close any open array
        if not truncated.rstrip().endswith("]"):
            truncated = truncated.rstrip().rstrip(",") + "]"
        # Ensure it starts with [
        arr_start = truncated.find("[")
        if arr_start == -1:
            return []
        truncated = truncated[arr_start:]
        try:
            parsed = json.loads(truncated)
            if isinstance(parsed, list):
                logger.warning(f"Repaired truncated JSON: salvaged {len(parsed)} facts")
                return parsed
        except json.JSONDecodeError:
            pass
        return []
