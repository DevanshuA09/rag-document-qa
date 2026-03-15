"""
test_ingestion.py — Standalone Phase 1 verification test.

Run with:
    python tests/test_ingestion.py

No pytest required.  The script downloads a small public PDF (or uses
any PDF you place in data/), runs the full ingestion pipeline, and
asserts every contract of Phase 1.

Place any PDF in the data/ directory and set PDF_PATH below, or let the
script download a sample automatically.
"""

import json
import logging
import os
import ssl
import sys
import urllib.request
from pathlib import Path

# ── Make sure the project root is on sys.path ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_ingestion")

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_PDF_URL = "https://arxiv.org/pdf/1706.03762"
DATA_DIR = ROOT / "data"
VECTORSTORE_DIR = ROOT / "vectorstore"
PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"

# Ensure we use the project-local vectorstore during tests.
os.environ.setdefault("VECTORSTORE_PATH", str(VECTORSTORE_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(label: str, condition: bool, detail: str = "") -> bool:
    """Print a pass/fail line and return the result."""
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def _download_sample_pdf() -> None:
    """Download the Attention Is All You Need PDF if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists():
        logger.info("Sample PDF already present at '%s'.", PDF_PATH)
        return

    print(f"\nDownloading sample PDF from:\n  {SAMPLE_PDF_URL}")
    print("(This may take a few seconds…)\n")
    try:
        # SSL verification is disabled here intentionally — this is a test
        # script only and the arxiv.org certificate can fail on some systems.
        # urlretrieve() doesn't accept a context arg; use urlopen() instead.
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(SAMPLE_PDF_URL, context=ssl_context) as resp:
            PDF_PATH.write_bytes(resp.read())
        logger.info("Downloaded sample PDF to '%s'.", PDF_PATH)
    except Exception as exc:
        print(
            f"\n⚠️  Could not download the sample PDF: {exc}\n"
            "Please place any PDF in the data/ directory and update PDF_PATH "
            "at the top of this script."
        )
        sys.exit(1)


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests() -> None:
    """Run all Phase 1 checks and print a final summary."""
    results: list[bool] = []

    # ── Setup ─────────────────────────────────────────────────────────────────
    _download_sample_pdf()

    from src.ingestion.pipeline import ingest_document, get_collection_name
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_pages, get_chunk_stats
    from src.ingestion.embedder import Embedder
    from src.ingestion.vectorstore import VectorStore

    print("\n" + "=" * 60)
    print("Phase 1 Ingestion Pipeline — Verification Tests")
    print("=" * 60)

    # ────────────────────────────────────────────────────────────────────────
    # 1. PDF parsing
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1] PDF Parsing")
    pages = parse_pdf(str(PDF_PATH))
    results.append(_check("Pages parsed > 0", len(pages) > 0, f"{len(pages)} pages"))
    results.append(
        _check(
            "Each page has text, page_number, source_filename, tables",
            all(
                "text" in p and "page_number" in p and "source_filename" in p and "tables" in p
                for p in pages
            ),
        )
    )
    results.append(
        _check(
            "page_number is 1-indexed",
            pages[0]["page_number"] == 1,
            f"first page_number = {pages[0]['page_number']}",
        )
    )
    results.append(
        _check(
            "source_filename is just the filename (no path)",
            "/" not in pages[0]["source_filename"],
            pages[0]["source_filename"],
        )
    )
    print(f"     Sample page_number: {pages[0]['page_number']}  |  text snippet: {pages[0]['text'][:80]!r}")

    # ────────────────────────────────────────────────────────────────────────
    # 2. Chunking
    # ────────────────────────────────────────────────────────────────────────
    print("\n[2] Chunking")
    chunks = chunk_pages(pages)
    stats = get_chunk_stats(chunks)
    results.append(_check("Chunks produced > 0", len(chunks) > 0, f"{len(chunks)} chunks"))
    results.append(
        _check(
            "Each chunk has required keys",
            all(
                {"text", "raw_text", "page_number", "source_filename", "chunk_id", "chunk_index", "chunk_type", "bbox"} <= c.keys()
                for c in chunks
            ),
        )
    )
    results.append(
        _check(
            "chunk_index is sequential starting from 0",
            [c["chunk_index"] for c in chunks[:5]] == list(range(min(5, len(chunks)))),
        )
    )
    results.append(
        _check(
            "No chunk shorter than 50 chars",
            all(len(c["text"]) >= 50 for c in chunks),
        )
    )
    results.append(_check("chunk_id values are unique", len({c["chunk_id"] for c in chunks}) == len(chunks)))

    print(f"     Chunk stats: {stats}")
    print(f"     Sample chunk (index 0):")
    sample = chunks[0]
    print(f"       chunk_id    : {sample['chunk_id']}")
    print(f"       page_number : {sample['page_number']}")
    print(f"       text snippet: {sample['text'][:100]!r}")

    # ────────────────────────────────────────────────────────────────────────
    # 3. Embedding (query only — full embed tested inside pipeline)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[3] Query Embedding")
    embedder = Embedder()
    test_query = "What are the accessibility success criteria?"
    query_vec = embedder.embed_query(test_query)
    results.append(
        _check("embed_query returns a non-empty list[float]", isinstance(query_vec, list) and len(query_vec) > 0)
    )
    results.append(
        _check(
            "Query embedding has correct type (list of floats)",
            all(isinstance(v, float) for v in query_vec[:5]),
        )
    )
    print(f"     Embedding dim: {len(query_vec)}")

    # ────────────────────────────────────────────────────────────────────────
    # 4. Full pipeline — first run (fresh ingestion)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[4] Full Pipeline — First Run")

    # Delete any leftover collection from a previous test run so we get a
    # deterministic "fresh ingestion" path.
    collection_name = get_collection_name(str(PDF_PATH))
    vs = VectorStore(persist_path=str(VECTORSTORE_DIR))
    if vs.collection_exists(collection_name):
        vs.delete_collection(collection_name)
        cache_file = VECTORSTORE_DIR / f"{collection_name}_chunks.json"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cleared previous test collection '%s'.", collection_name)

    result = ingest_document(str(PDF_PATH))
    results.append(_check("already_existed is False on first run", result["already_existed"] is False))
    results.append(_check("page_count > 0", result["page_count"] > 0, f"{result['page_count']} pages"))
    results.append(_check("chunk_count > 0", result["chunk_count"] > 0, f"{result['chunk_count']} chunks"))
    results.append(_check("collection_name is non-empty", bool(result["collection_name"])))
    results.append(_check("avg_chunk_length > 0", result["avg_chunk_length"] > 0))
    print(f"     Pipeline result: {result}")

    # ────────────────────────────────────────────────────────────────────────
    # 5. ChromaDB collection exists and is queryable
    # ────────────────────────────────────────────────────────────────────────
    print("\n[5] ChromaDB Collection")
    results.append(
        _check("collection_exists() returns True after ingestion", vs.collection_exists(collection_name))
    )

    col_stats = vs.get_collection_stats(collection_name)
    results.append(
        _check(
            "get_collection_stats() document_count matches chunk_count",
            col_stats["document_count"] == result["chunk_count"],
            f"{col_stats['document_count']} docs",
        )
    )

    # ────────────────────────────────────────────────────────────────────────
    # 6. ChromaDB query returns correct schema
    # ────────────────────────────────────────────────────────────────────────
    print("\n[6] ChromaDB Query")
    query_results = vs.query(query_vec, collection_name, top_k=5)
    results.append(_check("Query returns results", len(query_results) > 0, f"{len(query_results)} results"))

    required_keys = {"text", "page_number", "source_filename", "chunk_id", "chunk_index", "similarity_score", "chunk_type"}
    results.append(
        _check(
            "Result dicts have required schema",
            all(required_keys <= r.keys() for r in query_results),
        )
    )
    results.append(
        _check(
            "similarity_score is in [0, 1]",
            all(0.0 <= r["similarity_score"] <= 1.0 for r in query_results),
        )
    )
    results.append(
        _check(
            "Results sorted by similarity_score descending",
            query_results == sorted(query_results, key=lambda r: r["similarity_score"], reverse=True),
        )
    )
    print(f"     Top result (score={query_results[0]['similarity_score']:.4f}): {query_results[0]['text'][:80]!r}")

    # ────────────────────────────────────────────────────────────────────────
    # 7. Idempotency — second run must skip re-ingestion
    # ────────────────────────────────────────────────────────────────────────
    print("\n[7] Idempotency (second ingestion run)")
    result2 = ingest_document(str(PDF_PATH))
    results.append(
        _check("already_existed is True on second run", result2["already_existed"] is True)
    )
    results.append(
        _check(
            "chunk_count unchanged after second run",
            result2["chunk_count"] == result["chunk_count"],
            f"{result2['chunk_count']} chunks",
        )
    )

    # ────────────────────────────────────────────────────────────────────────
    # 8. BM25 side-car JSON exists
    # ────────────────────────────────────────────────────────────────────────
    print("\n[8] BM25 Chunk Cache")
    cache_path = VECTORSTORE_DIR / f"{collection_name}_chunks.json"
    results.append(_check("_chunks.json file exists", cache_path.exists(), str(cache_path)))

    if cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        results.append(_check("Cache has correct number of chunks", len(cached) == result["chunk_count"]))
        results.append(
            _check(
                "Cache does not contain embedding vectors",
                all("embedding" not in c for c in cached),
            )
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✅  Phase 1 complete. All {total} checks passed.")
    else:
        failed = total - passed
        print(f"❌  {failed}/{total} checks FAILED. Review the output above.")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_tests()
