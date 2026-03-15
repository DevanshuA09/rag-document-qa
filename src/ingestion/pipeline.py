"""
pipeline.py — Orchestration layer for document ingestion (Phase 1).

Wires together pdf_parser → chunker → embedder → vectorstore and handles
the BM25 chunk cache (JSON side-car file).

Primary interface:
    result = ingest_document(pdf_path)

Helper:
    collection_name = get_collection_name(pdf_path)
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .chunker import chunk_pages, get_chunk_stats
from .embedder import Embedder
from .pdf_parser import parse_pdf
from .vectorstore import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_collection_name(pdf_path: str) -> str:
    """Derive a stable, sanitised ChromaDB collection name from a PDF path.

    Example: ``"data/Apple 10-K.pdf"`` → ``"apple_10-k"``

    Rules:
    - Use the filename stem only (no directory, no extension)
    - Lowercase everything
    - Replace spaces with underscores
    - Strip any characters that are not alphanumeric, hyphens, or underscores

    Args:
        pdf_path: Full path or bare filename of the PDF.

    Returns:
        Sanitised collection name string (non-empty).
    """
    stem = Path(pdf_path).stem                          # e.g. "Apple 10-K"
    name = stem.lower()                                 # "apple 10-k"
    name = name.replace(" ", "_")                       # "apple_10-k"
    name = re.sub(r"[^\w\-]", "", name)                 # keep word chars + hyphens
    name = re.sub(r"_+", "_", name).strip("_")          # collapse multiple underscores
    if not name:
        name = "document"
    return name


def ingest_document(pdf_path: str) -> dict:
    """Run the full ingestion pipeline for a single PDF.

    Steps
    -----
    1. Derive ``collection_name`` from the filename.
    2. Check if the collection already exists in ChromaDB → skip if so.
    3. Parse the PDF into pages.
    4. Chunk pages.
    5. Embed chunks.
    6. Store in ChromaDB.
    7. Write a ``{collection_name}_chunks.json`` side-car for BM25 retrieval.
    8. Return a summary dict.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        Summary dict::

            {
                "collection_name":  str,
                "source_filename":  str,
                "page_count":       int,
                "chunk_count":      int,
                "already_existed":  bool,
                "avg_chunk_length": float,
                "vectorstore_path": str,
            }

    Raises:
        ValueError: Propagated from pdf_parser if the file is missing,
            encrypted, or corrupt.
    """
    pdf_path_obj = Path(pdf_path).expanduser().resolve()
    source_filename = pdf_path_obj.name
    collection_name = get_collection_name(str(pdf_path_obj))

    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    vs = VectorStore(persist_path=vectorstore_path)

    # ── Step 2: short-circuit if already ingested ────────────────────────────
    if vs.collection_exists(collection_name):
        logger.info(
            "Document '%s' already ingested (collection '%s'). Skipping.",
            source_filename,
            collection_name,
        )
        stats = vs.get_collection_stats(collection_name)

        # Derive page_count and avg_chunk_length from the JSON chunk cache so
        # the UI shows accurate numbers rather than 0.
        page_count = 0
        avg_chunk_length = 0.0
        cache_path = Path(vectorstore_path) / f"{collection_name}_chunks.json"
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as fh:
                cached_chunks = json.load(fh)
            page_count = len({c["page_number"] for c in cached_chunks})
            lengths = [len(c.get("raw_text", c["text"])) for c in cached_chunks]
            avg_chunk_length = round(sum(lengths) / len(lengths), 1) if lengths else 0.0

        return {
            "collection_name": collection_name,
            "source_filename": source_filename,
            "page_count": page_count,
            "chunk_count": stats["document_count"],
            "already_existed": True,
            "avg_chunk_length": avg_chunk_length,
            "vectorstore_path": stats["persist_path"],
        }

    # ── Step 3: Parse ────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Parsing PDF: %s", source_filename)
    pages = parse_pdf(str(pdf_path_obj))
    page_count = len(pages)

    # ── Step 4: Chunk ────────────────────────────────────────────────────────
    logger.info("Step 2/5 — Chunking %d pages.", page_count)
    chunks = chunk_pages(pages)
    chunk_stats = get_chunk_stats(chunks)

    # ── Step 5: Embed ────────────────────────────────────────────────────────
    logger.info("Step 3/5 — Embedding %d chunks.", len(chunks))
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)

    # ── Step 6: Store in ChromaDB ────────────────────────────────────────────
    logger.info("Step 4/5 — Storing in ChromaDB (collection: '%s').", collection_name)
    stored = vs.ingest_chunks(chunks, collection_name)

    # ── Step 7: Write BM25 side-car (chunks without embeddings) ─────────────
    logger.info("Step 5/5 — Writing BM25 chunk cache.")
    _write_chunk_cache(chunks, collection_name, vectorstore_path)

    stats = vs.get_collection_stats(collection_name)

    logger.info(
        "Ingestion complete: '%s' → %d pages, %d chunks stored.",
        source_filename,
        page_count,
        stored,
    )

    return {
        "collection_name": collection_name,
        "source_filename": source_filename,
        "page_count": page_count,
        "chunk_count": stored,
        "already_existed": False,
        "avg_chunk_length": chunk_stats["avg_chunk_length"],
        "vectorstore_path": stats["persist_path"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_chunk_cache(
    chunks: list[dict], collection_name: str, vectorstore_path: str
) -> Path:
    """Serialise chunks (without their embeddings) to a JSON side-car file.

    The BM25 index is rebuilt from this file at query time, so embeddings
    are excluded to keep the file small.

    Args:
        chunks: Embedded chunk dicts.
        collection_name: Used to name the output file.
        vectorstore_path: Directory where the file should be written.

    Returns:
        Path to the written JSON file.
    """
    cache_dir = Path(vectorstore_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{collection_name}_chunks.json"

    # Strip embedding vectors — they can be 100s of floats per chunk.
    lightweight = [
        {k: v for k, v in c.items() if k != "embedding"}
        for c in chunks
    ]

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(lightweight, f, ensure_ascii=False, indent=2)

    logger.info("BM25 chunk cache written to '%s'.", cache_path)
    return cache_path
