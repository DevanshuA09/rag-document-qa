"""
test_retrieval.py — Standalone Phase 2 verification test.

Run with:
    python tests/test_retrieval.py

Assumes Phase 1 has already been run and the following exist:
    - ChromaDB collection "attention_is_all_you_need"
    - vectorstore/attention_is_all_you_need_chunks.json

If they don't exist, the script will tell you to run Phase 1 first.
"""

import logging
import os
import sys
import time
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("VECTORSTORE_PATH", str(ROOT / "vectorstore"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_retrieval")

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "attention_is_all_you_need"
CHUNKS_JSON = ROOT / "vectorstore" / f"{COLLECTION_NAME}_chunks.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def _guard_phase1() -> None:
    """Exit with a clear message if Phase 1 artifacts are missing."""
    missing = []
    if not CHUNKS_JSON.exists():
        missing.append(f"  - {CHUNKS_JSON}")

    vectorstore_db = ROOT / "vectorstore" / "chroma.sqlite3"
    if not vectorstore_db.exists():
        missing.append(f"  - {vectorstore_db}")

    if missing:
        print("\n❌  Phase 1 artifacts not found:")
        for m in missing:
            print(m)
        print(
            "\nPlease run Phase 1 first:\n"
            "  .venv/bin/python3 tests/test_ingestion.py\n"
        )
        sys.exit(1)


# ── Test sections ─────────────────────────────────────────────────────────────

def test_bm25(results: list[bool]) -> list[dict]:
    """[1] BM25 Retriever."""
    print("\n[1] BM25 Retriever")
    from src.retrieval.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever.from_json(str(CHUNKS_JSON))
    hits = bm25.retrieve("attention mechanism transformer", top_k=5)

    results.append(_check("Returns 5 results", len(hits) == 5, f"{len(hits)} results"))
    required = {"text", "page_number", "chunk_id", "bm25_score"}
    results.append(
        _check(
            "Each result has text, page_number, chunk_id, bm25_score",
            all(required <= h.keys() for h in hits),
        )
    )
    results.append(
        _check(
            "Sorted by bm25_score descending",
            hits == sorted(hits, key=lambda h: h["bm25_score"], reverse=True),
        )
    )
    results.append(
        _check(
            "bm25_score values are floats >= 0",
            all(isinstance(h["bm25_score"], float) and h["bm25_score"] >= 0 for h in hits),
        )
    )
    print(f"     Top score: {hits[0]['bm25_score']:.4f}")
    print(f"     Snippet  : {hits[0]['text'][:100]!r}")
    return hits


def test_dense(results: list[bool]) -> list[dict]:
    """[2] Dense retrieval (ChromaDB only)."""
    print("\n[2] Dense Retrieval (ChromaDB)")
    from src.ingestion.embedder import Embedder
    from src.ingestion.vectorstore import VectorStore

    embedder = Embedder()
    vs = VectorStore()

    query = "what is multi-head attention?"
    embedding = embedder.embed_query(query)
    hits = vs.query(embedding, COLLECTION_NAME, top_k=5)

    results.append(_check("Returns 5 results", len(hits) == 5, f"{len(hits)} results"))
    results.append(
        _check(
            "similarity_score in [0, 1]",
            all(0.0 <= h["similarity_score"] <= 1.0 for h in hits),
        )
    )
    print(f"     Top score: {hits[0]['similarity_score']:.4f}")
    print(f"     Snippet  : {hits[0]['text'][:100]!r}")
    return hits


def test_hybrid(results: list[bool]) -> list[dict]:
    """[3] Hybrid retrieval (RRF fusion)."""
    print("\n[3] Hybrid Retrieval (RRF)")
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(COLLECTION_NAME, str(CHUNKS_JSON))
    query = "how does the transformer model work?"
    hits = retriever.retrieve(query, top_k=10)

    results.append(
        _check("Returns up to 10 results", 0 < len(hits) <= 10, f"{len(hits)} results")
    )
    required = {"rrf_score", "in_dense", "in_sparse", "dense_rank", "sparse_rank", "chunk_type"}
    results.append(
        _check(
            "Each result has rrf_score, in_dense, in_sparse, dense_rank, sparse_rank",
            all(required <= set(h.keys()) for h in hits),
        )
    )
    results.append(
        _check("rrf_score > 0 for all results", all(h["rrf_score"] > 0 for h in hits))
    )
    results.append(
        _check(
            "Sorted by rrf_score descending",
            hits == sorted(hits, key=lambda h: h["rrf_score"], reverse=True),
        )
    )

    # Fusion overlap stats.
    both = sum(1 for h in hits if h["in_dense"] and h["in_sparse"])
    dense_only = sum(1 for h in hits if h["in_dense"] and not h["in_sparse"])
    sparse_only = sum(1 for h in hits if h["in_sparse"] and not h["in_dense"])
    print(f"     Fusion overlap: {both} in-both, {dense_only} dense-only, {sparse_only} sparse-only")

    print("     Top 3 results:")
    for i, h in enumerate(hits[:3]):
        print(
            f"       [{i+1}] rrf={h['rrf_score']:.5f}  "
            f"dense={'✓' if h['in_dense'] else '✗'}(rank {h['dense_rank']})  "
            f"sparse={'✓' if h['in_sparse'] else '✗'}(rank {h['sparse_rank']})  "
            f"| {h['text'][:70]!r}"
        )
    return hits


def test_reranker(results: list[bool], hybrid_hits: list[dict]) -> list[dict]:
    """[4] Reranker."""
    print("\n[4] Reranker")
    from src.retrieval.reranker import Reranker

    reranker = Reranker()
    query = "how does the transformer model work?"
    reranked = reranker.rerank(query, hybrid_hits, top_k=5)

    results.append(
        _check("Returns exactly 5 results", len(reranked) == 5, f"{len(reranked)} results")
    )
    results.append(
        _check(
            "Each result has rerank_score (float)",
            all(isinstance(h.get("rerank_score"), float) for h in reranked),
        )
    )
    results.append(
        _check(
            "Sorted by rerank_score descending",
            reranked == sorted(reranked, key=lambda h: h["rerank_score"], reverse=True),
        )
    )
    print(f"     Top rerank_score: {reranked[0]['rerank_score']:.4f}")
    print(f"     Snippet         : {reranked[0]['text'][:100]!r}")
    return reranked


def test_full_pipeline(results: list[bool]) -> list[dict]:
    """[5] Full retrieval pipeline (end-to-end)."""
    print("\n[5] Full Retrieval Pipeline (end-to-end)")
    from src.retrieval.retrieval_pipeline import retrieve_chunks

    query = "what problem does self-attention solve?"
    chunks = retrieve_chunks(query, collection_name=COLLECTION_NAME)

    k_rerank = int(os.getenv("TOP_K_RERANK", "5"))
    results.append(
        _check(
            f"Returns up to {k_rerank} results",
            0 < len(chunks) <= k_rerank,
            f"{len(chunks)} results",
        )
    )
    required = {
        "text", "page_number", "source_filename", "chunk_id", "chunk_index",
        "rrf_score", "in_dense", "in_sparse", "dense_rank", "sparse_rank",
        "rerank_score", "chunk_type",
    }
    results.append(
        _check(
            "All required keys present",
            all(required <= set(c.keys()) for c in chunks),
        )
    )

    print("     Final retrieved chunks:")
    for i, c in enumerate(chunks):
        print(
            f"       [{i+1}] p{c['page_number']}  "
            f"rrf={c['rrf_score']:.5f}  rerank={c['rerank_score']:.4f}  "
            f"| {c['text'][:70]!r}"
        )
    return chunks


def test_cache(results: list[bool]) -> None:
    """[6] Cache check — second call must not reload models."""
    print("\n[6] Cache Check")
    from src.retrieval.retrieval_pipeline import retrieve_chunks

    query = "encoder decoder architecture"

    t0 = time.perf_counter()
    retrieve_chunks(query, collection_name=COLLECTION_NAME)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    retrieve_chunks(query, collection_name=COLLECTION_NAME)
    t_second = time.perf_counter() - t0

    results.append(
        _check(
            "Second call is faster (cached models)",
            t_second < t_first,
            f"1st={t_first:.2f}s  2nd={t_second:.2f}s",
        )
    )
    print(f"     1st call: {t_first:.3f}s  |  2nd call: {t_second:.3f}s")


def test_empty_query(results: list[bool]) -> None:
    """[7] Edge case — blank query handled gracefully."""
    print("\n[7] Edge Case — Empty Query")
    from src.retrieval.retrieval_pipeline import retrieve_chunks

    raised_correctly = False
    try:
        retrieve_chunks("", collection_name=COLLECTION_NAME)
    except ValueError as exc:
        raised_correctly = True
        print(f"     ValueError raised: {exc}")
    except Exception as exc:
        print(f"     Unexpected exception: {type(exc).__name__}: {exc}")

    results.append(
        _check(
            "Blank query raises ValueError (not a crash)",
            raised_correctly,
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    _guard_phase1()

    print("\n" + "=" * 60)
    print("Phase 2 Retrieval Layer — Verification Tests")
    print("=" * 60)

    all_results: list[bool] = []

    bm25_hits = test_bm25(all_results)
    _dense_hits = test_dense(all_results)
    hybrid_hits = test_hybrid(all_results)
    _reranked = test_reranker(all_results, hybrid_hits)
    _final = test_full_pipeline(all_results)
    test_cache(all_results)
    test_empty_query(all_results)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(all_results)
    total = len(all_results)
    if passed == total:
        print(f"✅  Phase 2 complete. All {total} checks passed.")
    else:
        failed = total - passed
        print(f"❌  {failed}/{total} checks FAILED. Review the output above.")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_tests()
