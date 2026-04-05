"""LoCoMo benchmark evaluation runner for MemForge.

Supports three evaluation metrics:
1. token_f1:  Token-level F1 with Porter stemming (LoCoMo official)
2. llm_judge: LLM-as-judge binary accuracy (ReMe-style, comparable to 86.23%)
3. both:      Report both metrics side by side

Usage:
    # Quick test, LLM judge (ReMe-comparable)
    python -m eval.locomo.run_eval --max-conversations 1 --metric llm_judge

    # Official LoCoMo F1
    python -m eval.locomo.run_eval --metric token_f1

    # Both metrics
    python -m eval.locomo.run_eval --metric both
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_f1_stemmed(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming (LoCoMo official method)."""
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
    except ImportError:
        stemmer = None

    def normalize(text: str) -> set[str]:
        # Lowercase, strip punctuation, split
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove stop words
        stop = {"a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
                "do", "did", "does", "has", "had", "have", "will", "would", "could",
                "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
                "my", "your", "his", "her", "its", "our", "their", "me", "him", "us",
                "them", "this", "that", "these", "those", "of", "in", "to", "for", "on",
                "at", "by", "with", "from", "as", "into", "about", "and", "or", "but",
                "so", "if", "than", "too", "very", "just", "not", "no", "yes"}
        tokens = [t for t in tokens if t not in stop]
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]
        return set(tokens)

    pred_tokens = normalize(prediction)
    gt_tokens = normalize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


LLM_JUDGE_PROMPT = """\
You are evaluating a question-answering system. Compare the generated answer with the gold answer.

## Rules
- If the generated answer captures the key information from the gold answer, mark it as CORRECT.
- Be generous: different wording, extra details, or partial matches are fine as long as the core fact is present.
- For time/date questions: allow format differences (e.g., "May 7, 2023" vs "2023-05-07" vs "7 May 2023").
- For unanswerable questions (gold answer is "unanswerable"): mark CORRECT only if the generated answer also indicates it cannot be answered.
- If the generated answer is completely wrong or irrelevant, mark it as WRONG.

## Question
{question}

## Gold Answer
{gold_answer}

## Generated Answer
{generated_answer}

## Verdict
Respond with ONLY a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


def llm_judge_score(
    question: str,
    prediction: str,
    ground_truth: str,
    judge_llm,
) -> float:
    """LLM-as-judge binary scoring (ReMe method). Returns 1.0 or 0.0."""
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=ground_truth,
        generated_answer=prediction,
    )
    try:
        response = judge_llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )
        # Parse verdict
        response = response.strip()
        if "CORRECT" in response.upper():
            return 1.0
        return 0.0
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return 0.0


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_locomo_dataset(path: str | None = None) -> list[dict]:
    p = Path(path) if path else LOCOMO_PATH
    if not p.exists():
        raise FileNotFoundError(f"LoCoMo not found at {p}. Clone: git clone https://github.com/snap-research/locomo")
    with open(p) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} conversations from {p}")
    return data


def flatten_conversation(conversation: dict) -> list[dict]:
    messages = []
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
            text = turn.get("text", "")
            if not text:
                continue
            speaker = turn.get("speaker", "")
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({
                "role": role,
                "content": f"[{session_date}] {speaker}: {text}" if session_date else f"{speaker}: {text}",
            })
    return messages


def get_observation_date(conversation: dict) -> str:
    date = conversation.get("session_1_date_time", "2023-01-01")
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return "2023-01-01"


# ── Main Evaluation ──────────────────────────────────────────────────────────


def run_memforge(
    dataset: list[dict],
    output_dir: Path,
    preset: str | None = None,
    config_path: str | None = None,
    max_workers: int = 8,
    metric: str = "both",
    skip_adversarial: bool = True,
    top_k: int = 20,
) -> dict:
    from memforge.core import MemForge
    from memforge.providers import BaseLLMClient, create_llm, get_preset
    from memforge.config import load_config

    # Load config to get per-role LLM settings
    if preset:
        preset_configs = get_preset(preset)
        qa_llm: BaseLLMClient = create_llm(preset_configs["llm"])
        judge_llm: BaseLLMClient = qa_llm
    else:
        cfg = load_config(config_path)
        # QA LLM: qa_llm > ingestion_llm
        qa_cfg = cfg.qa_llm or cfg.ingestion_llm
        qa_llm = create_llm(qa_cfg)
        # Judge LLM: judge_llm > qa_llm > ingestion_llm
        judge_cfg = cfg.judge_llm or qa_cfg
        judge_llm = create_llm(judge_cfg) if judge_cfg != qa_cfg else qa_llm

    use_f1 = metric in ("token_f1", "both")
    use_judge = metric in ("llm_judge", "both")

    results = []
    total_ingest_time = 0.0
    total_facts = 0
    type_scores: dict[str, dict[str, list[float]]] = {}  # {cat: {metric: [scores]}}

    output_dir.mkdir(parents=True, exist_ok=True)

    for conv_idx, item in enumerate(dataset):
        conv = item["conversation"]
        qas = item["qa"]
        sample_id = item.get("sample_id", conv_idx)
        logger.info(f"=== Conversation {conv_idx + 1}/{len(dataset)} (id={sample_id}, {len(qas)} questions) ===")

        messages = flatten_conversation(conv)
        observation_date = get_observation_date(conv)
        logger.info(f"  {len(messages)} messages, observation_date={observation_date}")

        # Create or reuse MemForge
        db_path = output_dir / f"conv_{conv_idx}.db"
        mf = MemForge(db_path=str(db_path), preset=preset, config_path=config_path)

        existing_count = mf.storage.count_facts()
        if existing_count > 0:
            total_facts += existing_count
            logger.info(f"  Reusing existing DB with {existing_count} facts")
        else:
            t0 = time.time()
            facts = mf.add(messages, session_id=f"conv_{conv_idx}", observation_date=observation_date)
            ingest_time = time.time() - t0
            total_ingest_time += ingest_time
            total_facts += len(facts)
            logger.info(f"  Ingested {len(facts)} facts in {ingest_time:.1f}s")

        # Filter QAs
        eval_qas = []
        for i, qa in enumerate(qas):
            category = qa.get("category", 0)
            if skip_adversarial and category == 5:
                continue
            eval_qas.append((i, qa))

        logger.info(f"  Evaluating {len(eval_qas)} questions (skip_adversarial={skip_adversarial})")

        # Parallel QA
        def answer_one_qa(qa_idx_qa):
            qa_idx, qa = qa_idx_qa
            question = qa["question"]
            ground_truth = qa.get("answer")
            category = qa.get("category", 0)
            cat_name = CATEGORY_NAMES.get(category, f"unknown_{category}")

            if ground_truth is None:
                ground_truth = "unanswerable"
            else:
                ground_truth = str(ground_truth)

            # Retrieve
            scored_facts = mf.search(question, top_k=top_k)
            context = "\n".join(sf.fact.content for sf in scored_facts)

            # Generate answer
            answer_prompt = (
                "Answer the following question based ONLY on the provided context.\n"
                "If the context doesn't contain enough information, say \"unanswerable\".\n"
                "Be concise. Answer in as few words as possible.\n\n"
                f"## Context\n{context if context else 'No relevant memories found.'}\n\n"
                f"## Question\n{question}\n\n"
                "## Answer"
            )
            prediction = qa_llm.chat(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=512,
            )

            # Compute metrics
            result = {
                "conversation_id": conv_idx,
                "sample_id": sample_id,
                "question_idx": qa_idx,
                "question": question,
                "question_type": cat_name,
                "category": category,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "num_facts_retrieved": len(scored_facts),
                "context_tokens": len(context.split()),
            }

            if use_f1:
                result["f1"] = compute_f1_stemmed(prediction, ground_truth)
            if use_judge:
                result["judge_score"] = llm_judge_score(question, prediction, ground_truth, judge_llm)

            return result

        from concurrent.futures import ThreadPoolExecutor, as_completed
        qa_results_for_conv = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(answer_one_qa, qa_tuple): qa_tuple[0] for qa_tuple in eval_qas}
            for future in as_completed(futures):
                try:
                    qa_results_for_conv.append(future.result())
                except Exception as e:
                    logger.error(f"  QA failed: {e}")

        qa_results_for_conv.sort(key=lambda x: x["question_idx"])
        for r in qa_results_for_conv:
            results.append(r)
            cat = r["question_type"]
            if cat not in type_scores:
                type_scores[cat] = {"f1": [], "judge": []}
            if "f1" in r:
                type_scores[cat]["f1"].append(r["f1"])
            if "judge_score" in r:
                type_scores[cat]["judge"].append(r["judge_score"])

        logger.info(f"  Answered {len(qa_results_for_conv)} questions")
        mf.close()

    # Aggregate
    total_q = len(results)

    summary = {
        "method": "memforge",
        "metric": metric,
        "skip_adversarial": skip_adversarial,
        "top_k": top_k,
        "total_questions": total_q,
        "total_facts": total_facts,
        "total_ingest_time_s": total_ingest_time,
    }

    logger.info("=" * 60)
    logger.info(f"MemForge LoCoMo Results (top_k={top_k}, skip_adv={skip_adversarial}):")
    logger.info(f"  Total questions: {total_q}")
    logger.info(f"  Total facts:    {total_facts}")

    if use_f1:
        all_f1 = [r["f1"] for r in results if "f1" in r]
        avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        summary["avg_f1"] = avg_f1
        summary["per_type_f1"] = {}
        logger.info(f"  ── Token F1 (stemmed, LoCoMo official) ──")
        logger.info(f"  Overall F1:     {avg_f1:.4f}")
        for cat in sorted(type_scores):
            scores = type_scores[cat]["f1"]
            if scores:
                avg = sum(scores) / len(scores)
                summary["per_type_f1"][cat] = avg
                logger.info(f"    {cat:15s}: F1={avg:.4f} (n={len(scores)})")

    if use_judge:
        all_judge = [r["judge_score"] for r in results if "judge_score" in r]
        avg_judge = sum(all_judge) / len(all_judge) if all_judge else 0
        summary["avg_judge_accuracy"] = avg_judge
        summary["per_type_judge"] = {}
        logger.info(f"  ── LLM Judge Accuracy (ReMe-comparable) ──")
        logger.info(f"  Overall Acc:    {avg_judge:.4f} ({avg_judge*100:.1f}%)")
        for cat in sorted(type_scores):
            scores = type_scores[cat]["judge"]
            if scores:
                avg = sum(scores) / len(scores)
                summary["per_type_judge"][cat] = avg
                logger.info(f"    {cat:15s}: Acc={avg:.4f} ({avg*100:.1f}%) (n={len(scores)})")

    logger.info("=" * 60)

    summary["results"] = results
    with open(output_dir / "memforge_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark Evaluation for MemForge")
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default=None,
                        choices=["openai", "anthropic", "google", "openrouter", "deepseek", "ollama"])
    parser.add_argument("--data", default=None, help="Path to locomo10.json")
    parser.add_argument("--output-dir", default="/tmp/memforge-eval")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metric", default="both", choices=["token_f1", "llm_judge", "both"],
                        help="Evaluation metric (default: both)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of facts to retrieve (default: 20)")
    parser.add_argument("--include-adversarial", action="store_true",
                        help="Include adversarial (category 5) questions")
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
        metric=args.metric,
        skip_adversarial=not args.include_adversarial,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
