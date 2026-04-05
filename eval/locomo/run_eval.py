"""LoCoMo benchmark evaluation runner for MemForge.

LoCoMo dataset structure (locomo10.json):
- 10 conversations, each with ~200 QA pairs
- Conversations have multiple sessions with timestamps
- QA categories: 1=single-hop, 2=multi-hop, 3=temporal, 4=open-ended, 5=adversarial

Usage:
    # Quick test (1 conversation)
    python -m eval.locomo.run_eval --max-conversations 1

    # Full eval (all 10)
    python -m eval.locomo.run_eval

    # With explicit config
    python -m eval.locomo.run_eval --config memforge.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOCOMO_PATH = Path(__file__).resolve().parents[2].parent / "locomo" / "data" / "locomo10.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_ended",
    5: "adversarial",
}


def load_locomo_dataset(path: str | None = None) -> list[dict]:
    """Load LoCoMo dataset from local JSON file."""
    p = Path(path) if path else LOCOMO_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {p}. "
            f"Clone it: git clone https://github.com/snap-research/locomo"
        )
    with open(p) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} conversations from {p}")
    return data


def flatten_conversation(conversation: dict) -> list[dict]:
    """Flatten multi-session conversation into a single message list with dates."""
    messages = []
    # Find all session keys
    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith(("_date_time", "_summary"))],
        key=lambda k: int(k.split("_")[1]),
    )
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_key = f"session_{session_num}_date_time"
        session_date = conversation.get(date_key, "")

        turns = conversation[session_key]
        if not isinstance(turns, list):
            continue

        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if not text:
                continue
            # Map speaker names to roles
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({
                "role": role,
                "content": f"[{session_date}] {speaker}: {text}" if session_date else f"{speaker}: {text}",
            })
    return messages


def get_observation_date(conversation: dict) -> str:
    """Get the first session date as observation anchor."""
    date = conversation.get("session_1_date_time", "2023-01-01")
    # Parse "May 7, 2023" style dates
    try:
        from datetime import datetime
        dt = datetime.strptime(date, "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(date, "%d %B %Y")
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return "2023-01-01"


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
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
    max_workers: int = 8,
) -> dict:
    """Run MemForge evaluation on LoCoMo."""
    from memforge.core import MemForge
    from memforge.providers import BaseLLMClient, create_llm, get_preset
    from memforge.config import load_config

    # Build QA answerer LLM
    if preset:
        qa_llm: BaseLLMClient = create_llm(get_preset(preset)["llm"])
    elif config_path:
        cfg = load_config(config_path)
        qa_llm = create_llm(cfg.llm)
    else:
        cfg = load_config()
        qa_llm = create_llm(cfg.llm)

    results = []
    total_f1 = 0.0
    total_questions = 0
    type_f1: dict[str, list[float]] = {}
    total_ingest_time = 0.0
    total_facts = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for conv_idx, item in enumerate(dataset):
        conv = item["conversation"]
        qas = item["qa"]
        sample_id = item.get("sample_id", conv_idx)
        logger.info(f"=== Conversation {conv_idx + 1}/{len(dataset)} (id={sample_id}, {len(qas)} questions) ===")

        # Flatten conversation
        messages = flatten_conversation(conv)
        observation_date = get_observation_date(conv)
        logger.info(f"  {len(messages)} messages, observation_date={observation_date}")

        # Create fresh MemForge per conversation
        db_path = output_dir / f"conv_{conv_idx}.db"
        if db_path.exists():
            db_path.unlink()
        mf = MemForge(db_path=str(db_path), preset=preset, config_path=config_path)

        # Ingest
        t0 = time.time()
        facts = mf.add(messages, session_id=f"conv_{conv_idx}", observation_date=observation_date)
        ingest_time = time.time() - t0
        total_ingest_time += ingest_time
        total_facts += len(facts)
        logger.info(f"  Ingested {len(facts)} facts in {ingest_time:.1f}s")

        # Answer questions (parallelized)
        def answer_one_qa(qa_idx_qa):
            qa_idx, qa = qa_idx_qa
            question = qa["question"]
            ground_truth = qa.get("answer")
            category = qa.get("category", 0)
            cat_name = CATEGORY_NAMES.get(category, f"unknown_{category}")

            # Handle None answers (adversarial/unanswerable) and int answers
            if ground_truth is None:
                ground_truth = "unanswerable"
            else:
                ground_truth = str(ground_truth)

            # Retrieve
            t0 = time.time()
            scored_facts = mf.search(question, top_k=10)
            retrieval_time = time.time() - t0

            context = "\n".join(sf.fact.content for sf in scored_facts)
            context_tokens = len(context.split())

            # QA with LLM
            answer_prompt = (
                "Answer the following question based ONLY on the provided context.\n"
                "If the context doesn't contain enough information, say \"unanswerable\".\n"
                "Be concise and specific. Answer in as few words as possible.\n\n"
                f"## Context\n{context if context else 'No relevant memories found.'}\n\n"
                f"## Question\n{question}\n\n"
                "## Answer"
            )
            prediction = qa_llm.chat(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            f1 = compute_f1(prediction, ground_truth)

            return {
                "conversation_id": conv_idx,
                "sample_id": sample_id,
                "question_idx": qa_idx,
                "question": question,
                "question_type": cat_name,
                "category": category,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "f1": f1,
                "num_facts_retrieved": len(scored_facts),
                "context_tokens": context_tokens,
                "retrieval_time_ms": retrieval_time * 1000,
            }

        from concurrent.futures import ThreadPoolExecutor, as_completed
        qa_results_for_conv = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(answer_one_qa, (i, qa)): i
                for i, qa in enumerate(qas)
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                    qa_results_for_conv.append(r)
                except Exception as e:
                    logger.error(f"  QA failed: {e}")

        # Sort by original index and accumulate
        qa_results_for_conv.sort(key=lambda x: x["question_idx"])
        for r in qa_results_for_conv:
            results.append(r)
            cat_name = r["question_type"]
            type_f1.setdefault(cat_name, []).append(r["f1"])
            total_f1 += r["f1"]
            total_questions += 1

        logger.info(f"  Answered {len(qa_results_for_conv)} questions")

        mf.close()

    # Summary
    avg_f1 = total_f1 / total_questions if total_questions else 0
    type_averages = {t: sum(s) / len(s) for t, s in type_f1.items()}

    summary = {
        "method": "memforge",
        "avg_f1": avg_f1,
        "total_questions": total_questions,
        "total_facts": total_facts,
        "total_ingest_time_s": total_ingest_time,
        "per_type_f1": type_averages,
        "results": results,
    }

    with open(output_dir / "memforge_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"MemForge LoCoMo Results:")
    logger.info(f"  Overall F1:     {avg_f1:.4f}")
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Total facts:    {total_facts}")
    logger.info(f"  Ingest time:    {total_ingest_time:.1f}s")
    for t, score in sorted(type_averages.items()):
        count = len(type_f1[t])
        logger.info(f"  {t:15s}: F1={score:.4f} (n={count})")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark Evaluation for MemForge")
    parser.add_argument("--config", default=None, help="Path to memforge.yaml")
    parser.add_argument(
        "--preset", default=None,
        choices=["openai", "anthropic", "google", "openrouter", "deepseek", "ollama"],
    )
    parser.add_argument("--data", default=None, help="Path to locomo10.json")
    parser.add_argument("--output-dir", default="eval/locomo/results")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Parallel QA workers (default: 8)")
    args = parser.parse_args()

    dataset = load_locomo_dataset(args.data)
    if args.max_conversations:
        dataset = dataset[:args.max_conversations]

    run_memforge(
        dataset,
        Path(args.output_dir),
        preset=args.preset,
        config_path=args.config,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
