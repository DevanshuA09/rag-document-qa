"""
rag_chain.py — Stuff and Reciprocal RAG chain implementations.

Stuff mode   — multi-query retrieval (original + 2 paraphrases) → dedup →
               top-8 chunks → single LLM call.
               Paraphrase generation uses gpt-4o-mini (Ma et al., 2023 /
               RAG-Fusion pattern) to widen lexical and semantic coverage
               without additional reranking cost.

Reciprocal   — LLM generates 4 sub-questions → 4 independent retrievals
               (top-8 each) → dedup + rerank filter → up to 8 final chunks →
               single LLM call.
               Based on the Şakar & Emekci (2025) Reciprocal RAG methodology.

When mode=None (or mode="auto"), **Query Classification → Adaptive Routing**
(NirDiamant RAG Techniques, 2024) is applied:
    Factual / Numerical  → stuff chain   (fast, precise look-up)
    Analytical / Comparative → reciprocal chain  (multi-hop synthesis)

Unified entry point:
    answer(query, collection_name, mode=None) -> dict

Individual chains:
    stuff_chain(query, collection_name)      -> dict
    reciprocal_chain(query, collection_name) -> dict
"""

import logging
import os

from dotenv import load_dotenv

from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import (
    build_reciprocal_final_prompt,
    build_reciprocal_subquestion_prompt,
    build_reciprocal_system_prompt,
    build_stuff_prompt,
)
from src.retrieval.retrieval_pipeline import retrieve_chunks

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level singleton — one LLMClient per process.
_llm_client: LLMClient | None = None
_classifier_llm_client: LLMClient | None = None


def _get_llm() -> LLMClient:
    """Return the shared GPT-4o LLMClient, creating it on first use."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _get_cheap_llm() -> LLMClient:
    """Return a gpt-4o-mini LLMClient for cheap auxiliary calls (paraphrases, classification)."""
    global _classifier_llm_client
    if _classifier_llm_client is None:
        _classifier_llm_client = LLMClient(
            model=os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")
        )
    return _classifier_llm_client


# ---------------------------------------------------------------------------
# Source formatting helper
# ---------------------------------------------------------------------------

def _build_sources(chunks: list[dict]) -> list[dict]:
    """Convert reranked chunks into the source list returned to callers."""
    return [
        {
            "page_number":     c["page_number"],
            "source_filename": c.get("source_filename", ""),
            "text_excerpt":    c.get("raw_text", c["text"])[:600],
            "chunk_id":        c["chunk_id"],
            "rerank_score":    c.get("rerank_score", 0.0),
            "chunk_type":      c.get("chunk_type", "text"),
            "bbox":            c.get("bbox"),
        }
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Multi-query helpers (RAG-Fusion pattern, Ma et al. 2023)
# ---------------------------------------------------------------------------

def _generate_query_variants(query: str, n: int = 2) -> list[str]:
    """Generate n alternative phrasings of *query* using gpt-4o-mini.

    Different phrasings hit different BM25 and dense retrieval paths,
    increasing the probability that the relevant chunk surfaces in at least
    one result list before deduplication.  This is the RAG-Fusion /
    multi-query retrieval pattern (Ma et al., 2023).

    Args:
        query: Original user question.
        n: Number of paraphrases to generate (default 2).

    Returns:
        List of up to *n* alternative phrasings.  Returns empty list on
        failure so the caller can fall back to single-query retrieval.
    """
    try:
        llm = _get_cheap_llm()
        raw = llm.generate(
            system_prompt=(
                "You are a query paraphrasing assistant. "
                "Output ONLY the paraphrased questions, one per line, no numbering or explanation."
            ),
            user_prompt=(
                f"Write {n} alternative phrasings of the following question that preserve "
                f"the exact same meaning but use different words or structure.\n\n"
                f"Question: {query}"
            ),
        ).strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.info("[multi-query] Generated %d variants for: %r", len(variants), query[:60])
        return variants[:n]
    except Exception as exc:
        logger.warning("[multi-query] Paraphrase generation failed (%s); using original only.", exc)
        return []


def _multi_retrieve(
    queries: list[str],
    collection_name: str,
    top_k_per_query: int = 8,
    final_top_k: int = 8,
) -> list[dict]:
    """Retrieve for each query, deduplicate by chunk_id, return top *final_top_k*.

    For each chunk that appears across multiple queries the version with the
    highest rerank_score is kept — identical deduplication logic to the
    Reciprocal chain.

    Args:
        queries: List of query strings (original + paraphrases).
        collection_name: ChromaDB collection name.
        top_k_per_query: Chunks retrieved per query before deduplication.
        final_top_k: Maximum chunks returned after deduplication.

    Returns:
        Deduplicated, rerank-score-sorted list of chunk dicts.
    """
    seen: dict[str, dict] = {}
    for q in queries:
        try:
            chunks = retrieve_chunks(q, collection_name, top_k_rerank=top_k_per_query)
            for c in chunks:
                cid = c["chunk_id"]
                if cid not in seen or c.get("rerank_score", 0.0) > seen[cid].get("rerank_score", 0.0):
                    seen[cid] = c
        except Exception as exc:
            logger.warning("[multi-query] Retrieval failed for query %r: %s", q[:60], exc)

    result = sorted(seen.values(), key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return result[:final_top_k]


# ---------------------------------------------------------------------------
# Stuff chain
# ---------------------------------------------------------------------------

def stuff_chain(query: str, collection_name: str) -> dict:
    """Multi-query RAG: retrieve for original + 2 paraphrases → dedup → single LLM call.

    Paraphrasing widens the retrieval net without adding a reranking step.
    The original query always participates; paraphrase failures degrade
    gracefully back to single-query retrieval.

    Args:
        query: User's question.
        collection_name: ChromaDB collection for the ingested document.

    Returns:
        Result dict (see module docstring for schema).
    """
    logger.info("[stuff] Generating query variants for: %r", query[:80])
    variants = _generate_query_variants(query, n=2)
    all_queries = [query] + variants

    logger.info("[stuff] Multi-query retrieval with %d queries.", len(all_queries))
    chunks = _multi_retrieve(all_queries, collection_name)

    system_prompt, user_prompt = build_stuff_prompt(query, chunks)

    logger.info("[stuff] Calling LLM (model=%s, chunks=%d).", _get_llm().model, len(chunks))
    usage = _get_llm().generate_with_usage(system_prompt, user_prompt)

    return {
        "answer":            usage["response"],
        "mode":              "stuff",
        "query":             query,
        "query_variants":    variants,
        "sources":           _build_sources(chunks),
        "chunks_retrieved":  len(chunks),
        "tokens_used":       usage["total_tokens"],
        "cost_usd":          usage["cost_usd"],
        "retrieval_mode":    "multi-query+hybrid+rerank",
    }


# ---------------------------------------------------------------------------
# Reciprocal chain
# ---------------------------------------------------------------------------

def reciprocal_chain(query: str, collection_name: str) -> dict:
    """Multi-retrieval RAG (Reciprocal RAG): expand query → retrieve per sub-question.

    Steps:
        1. LLM generates 4 focused sub-questions.
        2. retrieve_chunks() is called for each sub-question with top_k_rerank=8,
           widening the candidate pool from 4×5=20 to 4×8=32 before deduplication.
        3. All chunks are pooled and deduplicated by chunk_id (keep highest rerank_score).
        4. Pool is sorted by rerank_score descending, capped at 8.
        5. Chunks with rerank_score <= 0 are filtered out (noise guard).
           Falls back to top-5 if fewer than 2 survive.
        6. Final LLM call with the curated context.

    Args:
        query: User's original question.
        collection_name: ChromaDB collection for the ingested document.

    Returns:
        Result dict with Reciprocal-specific fields appended.
    """
    llm = _get_llm()

    # ── Step 1: Generate sub-questions ───────────────────────────────────────
    logger.info("[reciprocal] Generating sub-questions for: %r", query[:80])
    sub_q_raw = llm.generate(
        system_prompt=build_reciprocal_system_prompt(),
        user_prompt=build_reciprocal_subquestion_prompt(query),
    )
    sub_questions = _parse_sub_questions(sub_q_raw)
    logger.info("[reciprocal] Sub-questions: %s", sub_questions)

    # ── Step 2: Retrieve for each sub-question (top_k_rerank=8 each) ─────────
    all_chunks: list[dict] = []
    for i, sq in enumerate(sub_questions):
        logger.info(
            "[reciprocal] Retrieving for sub-question %d/%d: %r",
            i + 1, len(sub_questions), sq[:60],
        )
        try:
            sub_chunks = retrieve_chunks(sq, collection_name, top_k_rerank=8)
            all_chunks.extend(sub_chunks)
        except Exception as exc:
            logger.warning("[reciprocal] Sub-question %d retrieval failed: %s", i + 1, exc)

    total_before_dedup = len(all_chunks)

    # ── Step 3 & 4: Deduplicate by chunk_id, keep highest rerank_score ───────
    best: dict[str, dict] = {}
    for chunk in all_chunks:
        cid = chunk["chunk_id"]
        if cid not in best or chunk.get("rerank_score", 0.0) > best[cid].get("rerank_score", 0.0):
            best[cid] = chunk

    deduped = sorted(best.values(), key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:8]
    chunks_after_dedup = len(deduped)

    # ── Step 5: Relevance filter ──────────────────────────────────────────────
    filtered = [c for c in deduped if c.get("rerank_score", 0.0) > 0]
    if len(filtered) < 2:
        logger.warning(
            "[reciprocal] Score filter left %d chunks — falling back to top-5.", len(filtered)
        )
        filtered = deduped[:5]

    chunks_used = len(filtered)
    logger.info(
        "[reciprocal] Chunks: %d retrieved → %d deduped → %d used.",
        total_before_dedup,
        chunks_after_dedup,
        chunks_used,
    )

    # ── Step 6: Final LLM call ────────────────────────────────────────────────
    system_prompt, user_prompt = build_reciprocal_final_prompt(query, filtered)
    logger.info("[reciprocal] Calling LLM with %d chunks.", chunks_used)
    usage = llm.generate_with_usage(system_prompt, user_prompt)

    return {
        "answer":                    usage["response"],
        "mode":                      "reciprocal",
        "query":                     query,
        "sub_questions":             sub_questions,
        "sources":                   _build_sources(filtered),
        "chunks_retrieved":          chunks_used,
        "total_chunks_before_dedup": total_before_dedup,
        "chunks_after_dedup":        chunks_after_dedup,
        "chunks_used_in_prompt":     chunks_used,
        "tokens_used":               usage["total_tokens"],
        "cost_usd":                  usage["cost_usd"],
        "retrieval_mode":            "hybrid+rerank",
    }


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def answer(
    query: str,
    collection_name: str,
    mode: str | None = None,
) -> dict:
    """Route a query to the appropriate RAG chain.

    Args:
        query: User's question.
        collection_name: ChromaDB collection for the ingested document.
        mode: ``"stuff"``, ``"reciprocal"``, or ``"auto"``.
            Reads ``RAG_MODE`` from env if not provided; defaults to ``"auto"``.

    Returns:
        Result dict from the chosen chain.

    Raises:
        ValueError: If an unknown mode is specified.
    """
    resolved_mode = mode or os.getenv("RAG_MODE", "auto")
    logger.info("[answer] mode=%r  query=%r", resolved_mode, query[:80])

    if resolved_mode == "auto":
        from src.generation.query_classifier import classify_query, route_query
        query_type = classify_query(query)
        resolved_mode = route_query(query)
        logger.info(
            "[answer] Auto-routed to %r chain (query_type=%r).",
            resolved_mode,
            query_type,
        )
        result = (
            stuff_chain(query, collection_name)
            if resolved_mode == "stuff"
            else reciprocal_chain(query, collection_name)
        )
        result["query_type"] = query_type
        result["auto_routed"] = True
        return result

    elif resolved_mode == "stuff":
        return stuff_chain(query, collection_name)
    elif resolved_mode == "reciprocal":
        return reciprocal_chain(query, collection_name)
    else:
        raise ValueError(
            f"Unknown RAG mode: {resolved_mode!r}. "
            "Valid values are 'stuff', 'reciprocal', or 'auto'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_sub_questions(raw: str) -> list[str]:
    """Parse a numbered list of sub-questions from LLM output.

    Handles formats like ``"1. Question"`` or ``"1) Question"``.

    Returns:
        List of question strings, stripped of numbering. Falls back to
        splitting on newlines if parsing produces fewer than 2 items.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    questions: list[str] = []
    for line in lines:
        if line and line[0].isdigit():
            rest = line.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if rest:
                questions.append(rest)
        else:
            questions.append(line)

    if len(questions) < 2:
        questions = [l for l in lines if len(l) > 10]

    return questions[:4]
