"""
bm25_retriever.py — Sparse retrieval using BM25 (Okapi BM25 via rank_bm25).

BM25 scores chunks by lexical overlap with the query — it rewards exact keyword
matches and is complementary to dense semantic search.

The index is built in memory from the *_chunks.json cache written by the
ingestion pipeline.  No disk persistence is needed: rebuilding from the JSON
takes < 1 second for a 100-page document.

Primary interface:
    bm25 = BM25Retriever.from_json("vectorstore/my_doc_chunks.json")
    results = bm25.retrieve("attention mechanism", top_k=20)
"""

import copy
import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """In-memory BM25 index over a document's chunks.

    Args:
        chunks: List of chunk dicts (from the ``*_chunks.json`` cache).
            Each dict must have at least a ``"text"`` key.
    """

    def __init__(self, chunks: list[dict]) -> None:
        if not chunks:
            logger.warning("BM25Retriever initialised with an empty chunk list.")
            self._chunks: list[dict] = []
            self._index: BM25Okapi | None = None
            return

        self._chunks = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built: %d chunks indexed.", len(chunks))

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_path: str) -> "BM25Retriever":
        """Load chunks from a JSON cache file and return an initialised retriever.

        This is the primary way to create a :class:`BM25Retriever`.  The JSON
        file is the ``*_chunks.json`` side-car written by the ingestion pipeline.

        Args:
            json_path: Path to the ``*_chunks.json`` file.

        Returns:
            An initialised :class:`BM25Retriever`.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks JSON not found at '{json_path}'. "
                "Run ingest_document() for this PDF first."
            )
        with path.open(encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info("Loaded %d chunks from '%s'.", len(chunks), json_path)
        return cls(chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Return the top-k chunks ranked by BM25 score.

        Args:
            query: User's natural-language question.
            top_k: Maximum number of results to return.

        Returns:
            List of chunk dicts sorted by ``bm25_score`` descending.
            Each dict is a *copy* of the original — the index is not mutated.
            Returns an empty list if the index is empty or the query is blank.
        """
        if not self._chunks or self._index is None:
            logger.warning("BM25 retrieve called on an empty index.")
            return []

        if not query.strip():
            logger.warning("BM25 retrieve called with a blank query.")
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)  # ndarray, one score per chunk

        # Pair each chunk with its score and sort descending.
        scored = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, chunk in scored:
            result = copy.copy(chunk)        # shallow copy — never mutate originals
            result["bm25_score"] = float(score)
            results.append(result)

        top_score = results[0]["bm25_score"] if results else 0.0
        logger.info(
            "BM25 retrieve: query=%r  top_score=%.4f  returned %d results.",
            query[:60],
            top_score,
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, clean markdown noise, and split on whitespace.

        Handles two table-specific problems:
        - Pipe characters (|) from markdown table syntax are stripped so they
          don't pollute the vocabulary with high-frequency noise tokens.
        - Thousands separators in numbers ("383,285" → "383285") so financial
          figures are a single matchable token.

        Both the corpus and every query must be tokenised identically.
        """
        # Remove markdown table pipes and separator rows.
        cleaned = re.sub(r"[|\\]", " ", text)
        # Remove thousands separators inside numbers so 383,285 → 383285.
        cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", cleaned)
        # Collapse whitespace.
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.lower().split()
