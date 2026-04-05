"""LoCoMo benchmark evaluation runner for MemForge.

LoCoMo (Long Conversation Memory) evaluates memory systems on:
- Single-hop factual questions
- Multi-hop reasoning
- Temporal reasoning
- Open-ended questions
- Adversarial (unanswerable) questions

Dataset: https://huggingface.co/datasets/LoCoMo/LoCoMo
Paper: https://arxiv.org/abs/2402.10790

Usage:
    # With OpenAI
    python -m eval.locomo.run_eval --preset openai

    # With OpenRouter (access any model)
    python -m eval.locomo.run_eval --preset openrouter

    # With Anthropic
    python -m eval.locomo.run_eval --preset anthropic

    # With Google Gemini
    python -m eval.locomo.run_eval --preset google

    # Quick test (1 conversation)
    python -m eval.locomo.run_eval --preset openrouter --max-conversations 1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_locomo_dataset() -> list[dict]:
    """Load LoCoMo dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("LoCoMo/LoCoMo", split="test")
        return list(ds)
    except Exception as e:
        logger.error(f"Failed to load LoCoMo dataset: {e}")
        logger.info("Install with: pip install datasets")
        raise


def parse_conversation(conversation_data: list[dict]) -> list[dict]:
    """Parse LoCoMo conversation into {role, content} messages."""
    messages = []
    for turn in conversation_data:
        role = "user" if turn.get("speaker_id", 0) == 0 else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def run_memforge(
    dataset: list[dict],
    output_dir: Path,
    preset: str | None = None,
    config_path: str | None = None,
) -> dict:
    """MemForge: Ingest conversations, then answer from memory."""
    from memforge.core import MemForge
    from memforge.providers import BaseLLMClient, create_llm, get_preset

    # Build QA answerer from same config
    if config_path:
        from memforge.config import load_config
        cfg = load_config(config_path)
        qa_llm: BaseLLMClient = create_llm(cfg.llm)
    elif preset:
        preset_configs = get_preset(preset)
        qa_llm = create_llm(preset_configs["llm"])
    else:
        # Auto-discover config
        from memforge.config import load_config
        cfg = load_config()
        qa_llm = create_llm(cfg.llm)

    results = []
    total_f1 = 0.0
    total_questions = 0
    type_f1: dict[str, list[float]] = {}

    for conv_idx, conversation in enumerate(dataset):
        logger.info(f"Processing conversation {conv_idx + 1}/{len(dataset)}")

        # Create fresh MemForge instance per conversation
        db_path = output_dir / f"conv_{conv_idx}.db"
        mf = MemForge(db_path=str(db_path), preset=preset, config_path=config_path)

        # Step 1: Ingest conversation
        messages = parse_conversation(conversation["conversation"])
        observation_date = conversation.get("date", "2024-01-01")

        t0 = time.time()
        facts = mf.add(messages, session_id=f"conv_{conv_idx}", observation_date=observation_date)
        ingest_time = time.time() - t0
        logger.info(f"  Ingested {len(facts)} facts in {ingest_time:.1f}s")

        # Step 2: Answer questions from memory
        for qa in conversation["questions"]:
            question = qa["question"]
            ground_truth = qa["answer"]
            q_type = qa.get("type", "unknown")

            # Retrieve relevant memories
            t0 = time.time()
            scored_facts = mf.search(question, top_k=10)
            retrieval_time = time.time() - t0

            # Format context from retrieved facts
            context = "\n".join(sf.fact.content for sf in scored_facts)

            # Answer with LLM
            answer_prompt = f"""Answer the following question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't know" or "unanswerable".
Be concise and specific.

## Context
{context if context else "No relevant memories found."}

## Question
{question}

## Answer"""

            prediction = qa_llm.chat(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            f1 = compute_f1(prediction, ground_truth)

            results.append({
                "conversation_id": conv_idx,
                "question": question,
                "question_type": q_type,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "f1": f1,
                "num_facts_retrieved": len(scored_facts),
                "context_tokens": len(context.split()),
                "retrieval_time_ms": retrieval_time * 1000,
            })

            if q_type not in type_f1:
                type_f1[q_type] = []
            type_f1[q_type].append(f1)
            total_f1 += f1
            total_questions += 1

        mf.close()

    avg_f1 = total_f1 / total_questions if total_questions else 0
    type_averages = {t: sum(scores) / len(scores) for t, scores in type_f1.items()}

    summary = {
        "method": "memforge",
        "preset": preset,
        "avg_f1": avg_f1,
        "total_questions": total_questions,
        "per_type_f1": type_averages,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "memforge_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"MemForge F1: {avg_f1:.4f} ({total_questions} questions)")
    logger.info(f"Per-type F1: {json.dumps(type_averages, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark Evaluation for MemForge")
    parser.add_argument(
        "--config", default=None, type=str,
        help="Path to memforge.yaml config file (auto-discovered if not set)",
    )
    parser.add_argument(
        "--preset", default=None,
        choices=["openai", "anthropic", "google", "openrouter", "deepseek", "ollama"],
        help="Provider preset (overrides config file)",
    )
    parser.add_argument("--output-dir", default="eval/locomo/results", type=str)
    parser.add_argument(
        "--max-conversations", type=int, default=None,
        help="Limit number of conversations (for quick testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("Loading LoCoMo dataset...")
    dataset = load_locomo_dataset()
    if args.max_conversations:
        dataset = dataset[:args.max_conversations]
    logger.info(f"Loaded {len(dataset)} conversations")

    source = f"preset={args.preset}" if args.preset else f"config={args.config or 'auto'}"
    logger.info(f"=== Running MemForge ({source}) ===")
    run_memforge(dataset, output_dir, preset=args.preset, config_path=args.config)


if __name__ == "__main__":
    main()
