"""
test_generation.py — Standalone Phase 3 verification test.

Run with:
    .venv/bin/python3 tests/test_generation.py

Assumes Phase 1 already ran:
    - ChromaDB collection "attention_is_all_you_need" exists
    - vectorstore/attention_is_all_you_need_chunks.json exists

Requires OPENAI_API_KEY to be set in .env
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before any os.getenv() calls so API keys and paths are available.
load_dotenv(ROOT / ".env")

os.environ.setdefault("VECTORSTORE_PATH", str(ROOT / "vectorstore"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_generation")

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION = "attention_is_all_you_need"
CHUNKS_JSON = ROOT / "vectorstore" / f"{COLLECTION}_chunks.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def _guard_prerequisites() -> None:
    """Exit early if Phase 1 artifacts or API key are missing."""
    missing = []
    if not CHUNKS_JSON.exists():
        missing.append(f"  - {CHUNKS_JSON}  (run tests/test_ingestion.py first)")
    if not (ROOT / "vectorstore" / "chroma.sqlite3").exists():
        missing.append("  - ChromaDB not found  (run tests/test_ingestion.py first)")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        missing.append("  - OPENAI_API_KEY not set in .env")

    if missing:
        print("\n❌  Prerequisites missing:")
        for m in missing:
            print(m)
        sys.exit(1)


# ── Test sections ─────────────────────────────────────────────────────────────

def test_llm_client(results: list[bool]) -> None:
    """[1] LLM Client — basic call and usage metadata."""
    print("\n[1] LLM Client")
    from src.generation.llm_client import LLMClient

    client = LLMClient()
    result = client.generate_with_usage(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2? Reply with just the number.",
    )

    results.append(_check("Response is '4'", result["response"].strip() == "4", repr(result["response"])))
    results.append(
        _check(
            "All usage keys present",
            all(k in result for k in ("response","model","prompt_tokens","completion_tokens","total_tokens","cost_usd")),
        )
    )
    results.append(_check("prompt_tokens > 0",    result["prompt_tokens"] > 0,    str(result["prompt_tokens"])))
    results.append(_check("completion_tokens > 0", result["completion_tokens"] > 0, str(result["completion_tokens"])))
    results.append(_check("cost_usd > 0",          isinstance(result["cost_usd"], float) and result["cost_usd"] > 0))

    print(f"     tokens: {result['prompt_tokens']} in + {result['completion_tokens']} out = {result['total_tokens']} total")
    print(f"     cost:   ${result['cost_usd']:.6f}")


def test_prompt_builder(results: list[bool]) -> None:
    """[2] Prompt builder — Stuff mode."""
    print("\n[2] Prompt Builder — Stuff Mode")
    from src.generation.prompt_templates import build_stuff_prompt

    fake_chunks = [
        {"text": "The transformer uses multi-head attention.", "page_number": 1,
         "chunk_id": "aaa", "chunk_index": 0, "rerank_score": 1.0},
        {"text": "Positional encoding is added to the input embeddings.", "page_number": 2,
         "chunk_id": "bbb", "chunk_index": 1, "rerank_score": 0.9},
    ]
    query = "What encoding is used?"
    system, user = build_stuff_prompt(query, fake_chunks)

    results.append(
        _check(
            "System prompt contains abstention instruction",
            "I cannot find sufficient information" in system,
        )
    )
    results.append(_check("User prompt contains 'Page 1'", "Page 1" in user))
    results.append(_check("User prompt contains 'Page 2'", "Page 2" in user))
    results.append(_check("User prompt contains the query", query in user))
    results.append(
        _check(
            "System prompt contains citation instruction",
            "[Page X]" in system,
        )
    )


def test_stuff_answerable(results: list[bool]) -> dict:
    """[3] Stuff chain — answerable question."""
    print("\n[3] Stuff Chain — Answerable Question")
    from src.generation.rag_chain import stuff_chain

    query = "What is the main contribution of the transformer architecture?"
    result = stuff_chain(query, COLLECTION)

    results.append(_check("'answer' key present and non-trivial", len(result.get("answer", "")) > 50))
    results.append(_check("'sources' is a non-empty list",        isinstance(result["sources"], list) and len(result["sources"]) > 0))
    results.append(
        _check(
            "Each source has required keys",
            all(
                {"page_number","text_excerpt","chunk_id","rerank_score","chunk_type","source_filename"} <= set(s.keys())
                for s in result["sources"]
            ),
        )
    )
    results.append(_check("tokens_used > 0", result.get("tokens_used", 0) > 0))
    results.append(_check("mode == 'stuff'", result.get("mode") == "stuff"))

    print(f"\n     Answer:\n{result['answer']}\n")
    print("     Sources:")
    for s in result["sources"]:
        print(f"       p{s['page_number']} (rerank={s['rerank_score']:.4f}): {s['text_excerpt'][:80]!r}")
    return result


def test_stuff_unanswerable(results: list[bool]) -> None:
    """[4] Stuff chain — question outside document scope."""
    print("\n[4] Stuff Chain — Unanswerable Question")
    from src.generation.rag_chain import stuff_chain

    query = "What is the population of Singapore in 2024?"
    result = stuff_chain(query, COLLECTION)

    abstained = "I cannot find sufficient information" in result["answer"]
    results.append(_check("System correctly abstains on out-of-scope question", abstained))

    if abstained:
        print("     ✅ System correctly abstained on unanswerable question")
    else:
        print(f"     ⚠️  Unexpected answer: {result['answer'][:200]!r}")


def test_reciprocal(results: list[bool]) -> dict:
    """[5] Reciprocal chain."""
    print("\n[5] Reciprocal Chain")
    from src.generation.rag_chain import reciprocal_chain

    query = "How does the transformer handle long-range dependencies compared to RNNs?"
    result = reciprocal_chain(query, COLLECTION)

    results.append(
        _check("'sub_questions' has exactly 4 items", len(result.get("sub_questions", [])) == 4,
               str(len(result.get("sub_questions", []))))
    )
    results.append(_check("'chunks_retrieved' > 0",      result.get("chunks_retrieved", 0) > 0))
    results.append(_check("answer exists and non-trivial", len(result.get("answer", "")) > 50))
    results.append(_check("mode == 'reciprocal'",          result.get("mode") == "reciprocal"))
    results.append(
        _check(
            "total_chunks_before_dedup >= chunks_after_dedup",
            result.get("total_chunks_before_dedup", 0) >= result.get("chunks_after_dedup", 0),
        )
    )

    print("\n     Sub-questions generated:")
    for i, sq in enumerate(result.get("sub_questions", []), 1):
        print(f"       {i}. {sq}")
    print(f"\n     Answer:\n{result['answer']}\n")
    return result


def test_dispatcher(results: list[bool]) -> None:
    """[6] answer() dispatcher routes correctly."""
    print("\n[6] answer() Dispatcher")
    from src.generation.rag_chain import answer

    query = "What attention mechanism is used in the transformer?"

    r_stuff = answer(query, COLLECTION, mode="stuff")
    results.append(_check("mode='stuff' routes to stuff chain",       r_stuff.get("mode") == "stuff"))

    r_recip = answer(query, COLLECTION, mode="reciprocal")
    results.append(_check("mode='reciprocal' routes to reciprocal chain", r_recip.get("mode") == "reciprocal"))

    results.append(_check("Both return non-empty answers",
                          len(r_stuff.get("answer","")) > 0 and len(r_recip.get("answer","")) > 0))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    _guard_prerequisites()

    print("\n" + "=" * 60)
    print("Phase 3 Generation Layer — Verification Tests")
    print("=" * 60)

    all_results: list[bool] = []

    test_llm_client(all_results)
    test_prompt_builder(all_results)
    test_stuff_answerable(all_results)
    test_stuff_unanswerable(all_results)
    test_reciprocal(all_results)
    test_dispatcher(all_results)

    print("\n" + "=" * 60)
    passed = sum(all_results)
    total  = len(all_results)
    if passed == total:
        print(f"✅  Phase 3 complete. All {total} generation tests passed.")
    else:
        failed = total - passed
        print(f"❌  {failed}/{total} checks FAILED. Review output above.")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_tests()
