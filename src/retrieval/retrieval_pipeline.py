"""
retrieval_pipeline.py — Orchestrator: HybridRetriever → Reranker → Context Expansion.

Exposes a single function that takes a query + collection name and returns
the final reranked chunks ready for the generation stage.

Module-level caches mean HybridRetriever and Reranker are instantiated at most
once per collection per process — model loading (sentence-transformers) is slow
and must not happen on every query.

Implements **Context Window Expansion / RSE** (Retrieval with Surrounding Evidence,
NirDiamant RAG Techniques 2024): after reranking, each chunk is expanded with its
immediately adjacent neighbours (index n-1 and n+1) from the full chunk corpus.
This recovers context that straddles chunk boundaries and has been shown to
improve QA accuracy by ~42.6 % on the KITE benchmark.

Primary interface:
    chunks = retrieve_chunks(query, collection_name="attention_is_all_you_need")

Helper:
    path = get_chunks_json_path("attention_is_all_you_need")
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

load_dotenv()

logger = logging.getLogger(__name__)

# ── Module-level caches ────────────────────────────────────────────────────────
# Keyed by collection_name.  Populated lazily on first use.
_retriever_cache: dict[str, HybridRetriever] = {}
_reranker_cache: dict[str, Reranker] = {}
# Full chunk corpus keyed by collection_name, then by chunk_index.
# Used by Context Window Expansion to fetch neighbouring chunks.
_corpus_cache: dict[str, dict[int, dict]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_corpus(collection_name: str, chunks_json_path: str) -> dict[int, dict]:
    """Load and cache the full chunk corpus for a collection.

    Returns:
        Dict mapping ``chunk_index`` → chunk dict for fast neighbour lookup.
    """
    if collection_name not in _corpus_cache:
        with open(chunks_json_path, "r", encoding="utf-8") as fh:
            all_chunks: list[dict] = json.load(fh)
        _corpus_cache[collection_name] = {c["chunk_index"]: c for c in all_chunks}
        logger.debug(
            "Loaded corpus for '%s': %d chunks.", collection_name, len(all_chunks)
        )
    return _corpus_cache[collection_name]


def _expand_chunk(chunk: dict, corpus: dict[int, dict]) -> dict:
    """Return a copy of *chunk* with ``text`` expanded by neighbouring chunks.

    Table chunks are returned unchanged — tables are atomic and their content
    must not be mixed with adjacent prose to preserve exact numerical data.

    The expanded text is:
        [prev_text] + "\\n\\n" + chunk_text + "\\n\\n" + [next_text]

    The original contextual text (with header) is preserved for the expanded
    version so the header still applies.  ``raw_text`` is similarly expanded
    for display.  The original ``text`` is stored in ``core_text`` for
    transparency.

    Args:
        chunk: A reranked chunk dict.
        corpus: Full corpus dict (chunk_index → chunk dict).

    Returns:
        New chunk dict with expanded ``text`` and ``raw_text``.
    """
    # Tables are atomic — never expand them with neighbouring prose.
    if chunk.get("chunk_type") == "table":
        return chunk

    idx = chunk.get("chunk_index")
    if idx is None:
        return chunk  # can't expand without an index

    parts_text: list[str] = []
    parts_raw: list[str] = []

    prev = corpus.get(idx - 1)
    if prev and prev.get("page_number") == chunk.get("page_number"):
        # Only include same-page neighbours to avoid cross-section pollution.
        parts_text.append(prev.get("raw_text", prev.get("text", "")))
        parts_raw.append(prev.get("raw_text", prev.get("text", "")))

    parts_text.append(chunk["text"])
    parts_raw.append(chunk.get("raw_text", chunk["text"]))

    nxt = corpus.get(idx + 1)
    if nxt and nxt.get("page_number") == chunk.get("page_number"):
        parts_text.append(nxt.get("raw_text", nxt.get("text", "")))
        parts_raw.append(nxt.get("raw_text", nxt.get("text", "")))

    expanded = dict(chunk)
    expanded["core_text"] = chunk["text"]
    expanded["text"] = "\n\n".join(p for p in parts_text if p)
    expanded["raw_text"] = "\n\n".join(p for p in parts_raw if p)
    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chunks_json_path(collection_name: str) -> str:
    """Return the absolute path to the BM25 chunk cache for a collection.

    Args:
        collection_name: Sanitised collection name (e.g. ``"attention_is_all_you_need"``).

    Returns:
        String path: ``{VECTORSTORE_PATH}/{collection_name}_chunks.json``.
    """
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    return str(Path(vectorstore_path) / f"{collection_name}_chunks.json")


def retrieve_chunks(
    query: str,
    collection_name: str,
    top_k_retrieval: int | None = None,
    top_k_rerank: int | None = None,
) -> list[dict]:
    """Run hybrid retrieval + cross-encoder reranking for a query.

    Args:
        query: User's natural-language question.
        collection_name: ChromaDB collection for the ingested document.
        top_k_retrieval: Candidate pool size for hybrid retrieval.
            Reads ``TOP_K_RETRIEVAL`` from env; defaults to 20.
        top_k_rerank: Final chunks returned after reranking.
            Reads ``TOP_K_RERANK`` from env; defaults to 5.

    Returns:
        List of up to ``top_k_rerank`` chunk dicts, sorted by
        ``rerank_score`` descending.  Each dict contains::

            text, page_number, source_filename, chunk_id, chunk_index,
            rrf_score, in_dense, in_sparse, dense_rank, sparse_rank,
            rerank_score

    Raises:
        ValueError: If ``query`` is blank.
        FileNotFoundError: If the chunks JSON for ``collection_name`` doesn't
            exist (i.e. the document has not been ingested yet).
    """
    if not query.strip():
        raise ValueError(
            "retrieve_chunks() received a blank query. "
            "Please provide a non-empty question."
        )

    k_retrieval = top_k_retrieval or int(os.getenv("TOP_K_RETRIEVAL", "20"))
    k_rerank = top_k_rerank or int(os.getenv("TOP_K_RERANK", "5"))

    # ── Step 1: Resolve and validate the chunks JSON path ────────────────────
    chunks_json_path = get_chunks_json_path(collection_name)
    if not Path(chunks_json_path).exists():
        raise FileNotFoundError(
            f"Chunks JSON not found at '{chunks_json_path}'. "
            f"Run ingest_document() for this collection first."
        )

    # ── Step 2: Get or create HybridRetriever (cached per collection) ────────
    if collection_name not in _retriever_cache:
        logger.info(
            "Creating HybridRetriever for collection '%s' (first use).",
            collection_name,
        )
        _retriever_cache[collection_name] = HybridRetriever(
            collection_name=collection_name,
            chunks_json_path=chunks_json_path,
        )
    retriever = _retriever_cache[collection_name]

    # ── Step 3: Hybrid retrieval ─────────────────────────────────────────────
    logger.info(
        "Retrieving top-%d candidates for query: %r", k_retrieval, query[:80]
    )
    candidates = retriever.retrieve(query, top_k=k_retrieval)

    if not candidates:
        logger.warning("Hybrid retrieval returned 0 candidates.")
        return []

    # ── Step 4: Get or create Reranker (cached — one per process is fine) ────
    if collection_name not in _reranker_cache:
        logger.info("Creating Reranker for collection '%s' (first use).", collection_name)
        _reranker_cache[collection_name] = Reranker()
    reranker = _reranker_cache[collection_name]

    # ── Step 5: Rerank ───────────────────────────────────────────────────────
    logger.info("Reranking %d candidates → top %d.", len(candidates), k_rerank)
    reranked = reranker.rerank(query, candidates, top_k=k_rerank)

    # ── Step 6: Context Window Expansion (RSE, NirDiamant 2024) ─────────────
    # Fetch n±1 same-page neighbours for each reranked chunk and merge text.
    # This recovers split sentences / cross-boundary evidence at low extra cost.
    corpus = _load_corpus(collection_name, chunks_json_path)
    final_chunks = [_expand_chunk(c, corpus) for c in reranked]
    logger.info(
        "Context expansion applied to %d chunks.", len(final_chunks)
    )

    logger.info(
        "retrieve_chunks complete: %d final chunks for query %r.",
        len(final_chunks),
        query[:60],
    )
    return final_chunks
