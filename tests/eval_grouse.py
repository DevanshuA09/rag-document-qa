"""
eval_grouse.py — GroUSE evaluation framework for RAG output quality.

GroUSE (Grounded, Relevant, Useful, Succinct Evaluation) assesses RAG outputs
across four orthogonal dimensions, each scored 1–5 by GPT-4o acting as an
evaluator.  The framework is inspired by the ARES and RAGAS evaluation
methodologies but simplified for single-document RAG assessment.

Metrics
-------
1. Faithfulness (1–5)
   Are all claims in the answer grounded in the retrieved context?
   Score 5 = every statement traceable to a source passage.
   Score 1 = answer contains fabricated or contradicted claims.

2. Completeness (1–5)
   Does the answer address all facets of the question?
   Score 5 = full coverage of what was asked.
   Score 1 = partial or single-facet response.

3. Answer Relevancy (1–5)
   Is the answer directly on-topic for the question?
   Score 5 = laser-focused, no tangential content.
   Score 1 = generic or off-topic response.

4. Usefulness (1–5)
   Would a domain expert find the answer actionable / insightful?
   Score 5 = substantive, adds genuine value.
   Score 1 = vague or empty answer.

GroUSE Score = mean(Faithfulness, Completeness, Relevancy, Usefulness)

Usage
-----
    python -m tests.eval_grouse --collection attention_is_all_you_need

Or import and call directly:

    from tests.eval_grouse import evaluate_sample, evaluate_dataset
    result = evaluate_sample(query, answer, context_chunks, collection_name)
    dataset_results = evaluate_dataset("tests/eval_questions.json", collection_name)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.generation.llm_client import LLMClient
from src.generation.rag_chain import answer as rag_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("eval_grouse")

# ---------------------------------------------------------------------------
# Evaluator prompts
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are an impartial RAG evaluator. "
    "Score the answer on a scale of 1 (very poor) to 5 (excellent). "
    "Respond with ONLY a JSON object, no markdown, no explanation. "
    "Format: {\"score\": <int>, \"reason\": \"<one sentence>\"}"
)

_FAITHFULNESS_PROMPT = (
    "Evaluate FAITHFULNESS: Are all claims in the answer grounded in the context below?\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER: {answer}\n\n"
    "Score 5 if every statement is traceable to the context. "
    "Score 1 if the answer contains fabricated or contradicted claims."
)

_COMPLETENESS_PROMPT = (
    "Evaluate COMPLETENESS: Does the answer address all aspects of the question?\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER: {answer}\n\n"
    "Score 5 if the answer fully covers what was asked. "
    "Score 1 if major aspects are missing."
)

_RELEVANCY_PROMPT = (
    "Evaluate ANSWER RELEVANCY: Is the answer directly on-topic and focused on the question?\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER: {answer}\n\n"
    "Score 5 if the answer is laser-focused with no tangential content. "
    "Score 1 if it is generic or off-topic."
)

_USEFULNESS_PROMPT = (
    "Evaluate USEFULNESS: Would a domain expert find this answer actionable or insightful?\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER: {answer}\n\n"
    "Score 5 if the answer is substantive and adds genuine value. "
    "Score 1 if it is vague, empty, or trivially obvious."
)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _score_metric(
    llm: LLMClient,
    prompt: str,
    question: str,
    answer: str,
    context: str = "",
) -> dict:
    """Call the evaluator LLM and parse the score/reason JSON.

    Args:
        llm: LLMClient instance (gpt-4o recommended for evaluation).
        prompt: Metric-specific prompt template with {question}/{answer}/{context}.
        question: User's question.
        answer: RAG-generated answer.
        context: Concatenated retrieved chunks (only needed for Faithfulness/Completeness).

    Returns:
        Dict with ``score`` (int 1-5) and ``reason`` (str).
    """
    filled = prompt.format(question=question, answer=answer, context=context)
    raw = llm.generate(_EVAL_SYSTEM, filled).strip()

    try:
        # Strip accidental markdown fences
        clean = raw.strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        parsed = json.loads(clean)
        score = max(1, min(5, int(parsed.get("score", 3))))
        reason = str(parsed.get("reason", "")).strip()
        return {"score": score, "reason": reason}
    except Exception as exc:
        logger.warning("Failed to parse evaluator output (%s): %r", exc, raw[:100])
        return {"score": 3, "reason": f"Parse error: {exc}"}


def evaluate_sample(
    query: str,
    answer_text: str,
    context_chunks: list[dict],
    evaluator_llm: Optional[LLMClient] = None,
) -> dict:
    """Compute GroUSE scores for a single (query, answer, context) triple.

    Args:
        query: The user's question.
        answer_text: The RAG-generated answer string.
        context_chunks: List of retrieved chunk dicts (must have ``"text"`` key).
        evaluator_llm: LLMClient to use for evaluation.  Defaults to GPT-4o.

    Returns:
        Dict with per-metric scores and the composite GroUSE score::

            {
                "faithfulness":    {"score": int, "reason": str},
                "completeness":    {"score": int, "reason": str},
                "answer_relevancy":{"score": int, "reason": str},
                "usefulness":      {"score": int, "reason": str},
                "grouse_score":    float,   # mean of all 4, rounded to 2dp
            }
    """
    if evaluator_llm is None:
        evaluator_llm = LLMClient(model=os.getenv("EVAL_MODEL", "gpt-4o"))

    # Build context string from whatever text key is available.
    # sources dicts use "text_excerpt"; full chunk dicts use "raw_text" or "text".
    context = "\n\n---\n\n".join(
        c.get("raw_text", c.get("text", c.get("text_excerpt", "")))[:1500]
        for c in context_chunks
    )

    faith = _score_metric(evaluator_llm, _FAITHFULNESS_PROMPT, query, answer_text, context)
    comp  = _score_metric(evaluator_llm, _COMPLETENESS_PROMPT, query, answer_text, context)
    relev = _score_metric(evaluator_llm, _RELEVANCY_PROMPT,    query, answer_text)
    use   = _score_metric(evaluator_llm, _USEFULNESS_PROMPT,   query, answer_text)

    grouse = round((faith["score"] + comp["score"] + relev["score"] + use["score"]) / 4, 2)

    return {
        "faithfulness":     faith,
        "completeness":     comp,
        "answer_relevancy": relev,
        "usefulness":       use,
        "grouse_score":     grouse,
    }


def evaluate_and_print(
    query: str,
    collection_name: str,
    mode: str = "auto",
    evaluator_llm: Optional[LLMClient] = None,
) -> dict:
    """Run the full RAG pipeline then evaluate the output.

    This is the top-level convenience function: it calls
    :func:`~src.generation.rag_chain.answer`, extracts the answer and source
    chunks, then calls :func:`evaluate_sample`.

    Args:
        query: User's question.
        collection_name: ChromaDB collection for the ingested document.
        mode: RAG chain mode (``"stuff"``, ``"reciprocal"``, ``"auto"``).
        evaluator_llm: Optional evaluator LLMClient.

    Returns:
        Combined dict with RAG result fields + GroUSE evaluation fields.
    """
    logger.info("Running RAG pipeline (mode=%r) for: %r", mode, query[:80])
    t0 = time.perf_counter()
    rag_result = rag_answer(query=query, collection_name=collection_name, mode=mode)
    elapsed = time.perf_counter() - t0

    answer_text = rag_result["answer"]
    # Build context chunks from the returned sources (text_excerpt is truncated;
    # use raw source metadata for the evaluator).
    context_chunks = rag_result.get("sources", [])

    logger.info("Evaluating answer with GroUSE metrics...")
    eval_result = evaluate_sample(
        query=query,
        answer_text=answer_text,
        context_chunks=context_chunks,
        evaluator_llm=evaluator_llm,
    )

    combined = {
        "query":           query,
        "mode":            rag_result.get("mode"),
        "query_type":      rag_result.get("query_type"),
        "answer":          answer_text,
        "rag_latency_s":   round(elapsed, 2),
        "tokens_used":     rag_result.get("tokens_used", 0),
        "cost_usd":        rag_result.get("cost_usd", 0.0),
        **eval_result,
    }

    _print_result(combined)
    return combined


def evaluate_dataset(
    questions_path: str,
    collection_name: str,
    mode: str = "auto",
    output_path: Optional[str] = None,
) -> list[dict]:
    """Evaluate a JSON file of questions against a collection.

    The questions file must be a JSON array of strings or objects with a
    ``"query"`` key.  Example::

        [
          "What is the company's main revenue source?",
          {"query": "How did net income change YoY?", "mode": "reciprocal"}
        ]

    Args:
        questions_path: Path to the JSON questions file.
        collection_name: ChromaDB collection to evaluate against.
        mode: Default RAG mode (can be overridden per question).
        output_path: If provided, results are saved as JSON here.

    Returns:
        List of result dicts from :func:`evaluate_and_print`.
    """
    with open(questions_path, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    evaluator_llm = LLMClient(model=os.getenv("EVAL_MODEL", "gpt-4o"))
    results = []

    for i, item in enumerate(questions, 1):
        if isinstance(item, str):
            q, q_mode = item, mode
        else:
            q, q_mode = item["query"], item.get("mode", mode)

        logger.info("=== Question %d/%d ===", i, len(questions))
        try:
            result = evaluate_and_print(
                query=q,
                collection_name=collection_name,
                mode=q_mode,
                evaluator_llm=evaluator_llm,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Question %d failed: %s", i, exc)
            results.append({"query": q, "error": str(exc), "grouse_score": None})

    # Summary
    valid = [r for r in results if r.get("grouse_score") is not None]
    if valid:
        avg_grouse = round(sum(r["grouse_score"] for r in valid) / len(valid), 2)
        avg_faith  = round(sum(r["faithfulness"]["score"] for r in valid) / len(valid), 2)
        avg_comp   = round(sum(r["completeness"]["score"] for r in valid) / len(valid), 2)
        avg_relev  = round(sum(r["answer_relevancy"]["score"] for r in valid) / len(valid), 2)
        avg_use    = round(sum(r["usefulness"]["score"] for r in valid) / len(valid), 2)
        print("\n" + "=" * 60)
        print(f"  DATASET GroUSE SUMMARY  ({len(valid)}/{len(results)} questions)")
        print("=" * 60)
        print(f"  GroUSE Score     : {avg_grouse} / 5.0")
        print(f"  Faithfulness     : {avg_faith}")
        print(f"  Completeness     : {avg_comp}")
        print(f"  Answer Relevancy : {avg_relev}")
        print(f"  Usefulness       : {avg_use}")
        print("=" * 60)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_result(r: dict) -> None:
    """Print a formatted GroUSE evaluation result to stdout."""
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  Query  : {r['query'][:70]}")
    print(f"  Mode   : {r.get('mode', '?')}  |  Type: {r.get('query_type', 'N/A')}")
    print(bar)
    print(f"  Answer : {r['answer'][:200]}{'…' if len(r['answer']) > 200 else ''}")
    print(bar)
    print("  GroUSE Evaluation")
    print(f"    Faithfulness     : {r['faithfulness']['score']}/5  — {r['faithfulness']['reason']}")
    print(f"    Completeness     : {r['completeness']['score']}/5  — {r['completeness']['reason']}")
    print(f"    Answer Relevancy : {r['answer_relevancy']['score']}/5  — {r['answer_relevancy']['reason']}")
    print(f"    Usefulness       : {r['usefulness']['score']}/5  — {r['usefulness']['reason']}")
    print(f"    ► GroUSE Score   : {r['grouse_score']} / 5.0")
    print(f"  Cost   : ${r.get('cost_usd', 0):.4f}  |  Tokens: {r.get('tokens_used', 0):,}")
    print(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroUSE evaluation for RAG outputs")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name")
    parser.add_argument("--query", help="Single query to evaluate")
    parser.add_argument("--questions", help="Path to JSON questions file for batch eval")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "stuff", "reciprocal"],
        help="RAG chain mode (default: auto)",
    )
    parser.add_argument("--output", help="Path to save JSON results (batch only)")

    args = parser.parse_args()

    if args.query:
        evaluate_and_print(
            query=args.query,
            collection_name=args.collection,
            mode=args.mode,
        )
    elif args.questions:
        evaluate_dataset(
            questions_path=args.questions,
            collection_name=args.collection,
            mode=args.mode,
            output_path=args.output,
        )
    else:
        parser.error("Provide either --query or --questions.")
