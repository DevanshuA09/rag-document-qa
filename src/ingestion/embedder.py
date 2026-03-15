"""
embedder.py — Dense embeddings via sentence-transformers (BAAI/bge-small-en-v1.5).

The model is loaded lazily on first use so import is always fast.
BGE retrieval mode requires a special query prefix; :meth:`Embedder.embed_query`
applies it automatically.

Primary interface:
    embedder = Embedder()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    query_vec = embedder.embed_query("What are the risks?")
"""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Recommended prefix for bge models in asymmetric retrieval.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_BATCH_SIZE = 64


class Embedder:
    """Wraps a SentenceTransformer model for document and query embedding.

    The underlying model is loaded on the first call to :meth:`embed_chunks`
    or :meth:`embed_query` — not at instantiation time — to keep startup fast.

    Args:
        model_name: HuggingFace model ID.  Reads ``EMBEDDING_MODEL`` from the
            environment if not provided; falls back to ``BAAI/bge-small-en-v1.5``.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = (
            model_name
            or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        )
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Embed each chunk's text and attach the result as an ``"embedding"`` key.

        Processes chunks in batches of 64 to avoid GPU/CPU memory pressure on
        large documents.  Modifies the input dicts in place **and** returns them
        so the caller can chain calls.

        Args:
            chunks: List of chunk dicts (from :func:`chunker.chunk_pages`).
                Must have a ``"text"`` key.

        Returns:
            The same list of chunk dicts, each now containing an
            ``"embedding"`` key with a ``list[float]`` value.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("embed_chunks received an empty chunk list.")

        model = self._load_model()
        texts = [c["text"] for c in chunks]
        total = len(texts)

        logger.info("Embedding %d chunks in batches of %d...", total, _BATCH_SIZE)
        t0 = time.perf_counter()

        all_embeddings: list[list[float]] = []
        num_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE

        for batch_idx in range(num_batches):
            start = batch_idx * _BATCH_SIZE
            end = min(start + _BATCH_SIZE, total)
            batch_texts = texts[start:end]

            batch_vecs = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.extend(batch_vecs.tolist())

            # Log progress every 10 batches (or always if tqdm is unavailable).
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                logger.info(
                    "  Embedded batch %d/%d (%d/%d chunks).",
                    batch_idx + 1,
                    num_batches,
                    end,
                    total,
                )

        elapsed = time.perf_counter() - t0
        cps = total / elapsed if elapsed > 0 else float("inf")
        logger.info(
            "Embedding complete: %d chunks in %.2fs (%.1f chunks/sec).",
            total,
            elapsed,
            cps,
        )

        for chunk, embedding in zip(chunks, all_embeddings):
            chunk["embedding"] = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval.

        Prepends the BGE query prefix before encoding, which is required for
        the asymmetric retrieval use-case (query vs. passage).

        Args:
            query: Raw user question string.

        Returns:
            Normalized embedding as a ``list[float]``.

        Raises:
            ValueError: If ``query`` is blank.
        """
        if not query.strip():
            raise ValueError("embed_query received a blank query string.")

        model = self._load_model()
        prefixed = _BGE_QUERY_PREFIX + query.strip()
        vec = model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the SentenceTransformer model on first access (lazy init).

        Returns:
            A loaded :class:`~sentence_transformers.SentenceTransformer` instance.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # local import keeps startup fast
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s'...", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        logger.info("Embedding model '%s' loaded successfully.", self.model_name)
        return self._model
