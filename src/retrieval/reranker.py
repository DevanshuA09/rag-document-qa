"""
reranker.py — Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.

A cross-encoder reads the *concatenation* of (query, chunk) together, giving
it full attention between both texts.  This makes it far more accurate than
the bi-encoder used for retrieval, at the cost of being O(n) — unsuitable for
full-corpus search but ideal for rescoring a small candidate set (≤ 40 chunks).

Implements **Relevance Threshold Filter** (inspired by CRAG, Yan et al. 2024):
chunks with a cross-encoder score below a configurable threshold are discarded
as irrelevant noise.  A minimum floor of 2 chunks is enforced so the LLM
always receives some context.  Default threshold: -3.0 (empirically safe —
only genuinely off-topic chunks score below this).

Primary interface:
    reranker = Reranker()
    final_chunks = reranker.rerank(query, candidates, top_k=5)
"""

import copy
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for a small set of retrieval candidates.

    Args:
        model_name: HuggingFace cross-encoder model ID.
            Reads ``RERANKER_MODEL`` from env if not provided;
            falls back to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = (
            model_name
            or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self._model = None  # lazy-loaded on first rerank() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        """Score and re-sort retrieval candidates using a cross-encoder.

        All (query, chunk) pairs are scored in a single batched forward pass
        for efficiency.  After sorting, a **relevance threshold filter** drops
        chunks whose score falls below *threshold* (CRAG-style noise removal),
        while guaranteeing at least 2 chunks are returned.

        Args:
            query: User's natural-language question.
            chunks: Candidate chunk dicts from :class:`HybridRetriever`.
                Must have a ``"text"`` key.
            top_k: How many top-scoring chunks to return before thresholding.
            threshold: Minimum acceptable rerank score.  Reads
                ``RERANK_THRESHOLD`` env var if not provided; defaults to
                ``-3.0``.  Set to ``None`` to disable filtering.

        Returns:
            Sorted list of chunk dicts (best first), each with a
            ``"rerank_score"`` (float) key added.  Input dicts are *not*
            mutated — copies are returned.

        Raises:
            ValueError: If ``chunks`` is empty or ``query`` is blank.
        """
        if not chunks:
            raise ValueError("rerank() received an empty candidate list.")
        if not query.strip():
            raise ValueError("rerank() received a blank query.")

        # Resolve threshold: argument > env var > default -3.0
        if threshold is None:
            threshold = float(os.getenv("RERANK_THRESHOLD", "-3.0"))

        model = self._load_model()
        pairs = [(query, c["text"]) for c in chunks]

        logger.debug("Reranking %d candidates...", len(pairs))
        t0 = time.perf_counter()

        scores = model.predict(pairs, show_progress_bar=False)

        elapsed = time.perf_counter() - t0

        # Pair original chunks with their scores, sort descending.
        all_scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # Standard top-k by cross-encoder score.
        top_scored = all_scored[:top_k]
        top_ids = {chunk["chunk_id"] for _, chunk in top_scored}

        # Table guarantee: table chunks that ranked in BM25 top-5 but fell
        # outside the top-k cross-encoder cut are injected here.  ms-marco was
        # trained on natural-language passages and systematically under-scores
        # pipe-separated markdown tables even when they contain the exact answer.
        # The threshold filter already exempts tables; this ensures they are
        # present in `results` for that exemption to apply.
        _TABLE_INJECT_MAX = 2
        injected_count = 0
        extra: list[tuple] = []
        for score, chunk in all_scored[top_k:]:
            if injected_count >= _TABLE_INJECT_MAX:
                break
            sparse_rank = chunk.get("sparse_rank")
            if (
                chunk.get("chunk_type") == "table"
                and sparse_rank is not None
                and sparse_rank <= 5
                and chunk["chunk_id"] not in top_ids
            ):
                extra.append((score, chunk))
                top_ids.add(chunk["chunk_id"])
                injected_count += 1
                logger.info(
                    "Reranker table injection: chunk_id=%s sparse_rank=%d rerank_score=%.4f.",
                    chunk["chunk_id"],
                    sparse_rank,
                    float(score),
                )

        scored = top_scored + extra

        results = []
        for score, chunk in scored:
            result = copy.copy(chunk)
            result["rerank_score"] = float(score)
            results.append(result)

        # ── Relevance Threshold Filter (CRAG-inspired, Yan et al. 2024) ───────
        # Discard chunks that are clearly off-topic (score < threshold).
        # Table chunks are exempt from the threshold — ms-marco was trained on
        # natural-language passages and consistently underscores markdown tables
        # even when they contain the exact answer.  We let the LLM decide.
        _MIN_CHUNKS = 2
        above = [
            r for r in results
            if r["rerank_score"] >= threshold or r.get("chunk_type") == "table"
        ]
        if len(above) >= _MIN_CHUNKS:
            filtered = above
            n_dropped = len(results) - len(filtered)
            if n_dropped:
                logger.info(
                    "Threshold filter (score < %.1f) dropped %d/%d chunks.",
                    threshold,
                    n_dropped,
                    len(results),
                )
        else:
            # Not enough chunks above threshold — keep top _MIN_CHUNKS anyway.
            filtered = results[:_MIN_CHUNKS]
            logger.warning(
                "Threshold filter would leave < %d chunks; "
                "keeping top %d (scores: %s).",
                _MIN_CHUNKS,
                _MIN_CHUNKS,
                [round(r["rerank_score"], 3) for r in filtered],
            )

        top_score = filtered[0]["rerank_score"] if filtered else float("nan")
        logger.info(
            "Reranked %d candidates → %d kept in %.3fs  (top_score=%.4f).",
            len(chunks),
            len(filtered),
            elapsed,
            top_score,
        )
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the CrossEncoder model on first access (lazy init).

        Returns:
            A loaded :class:`~sentence_transformers.CrossEncoder` instance.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading reranker model '%s'...", self.model_name)
        try:
            self._model = CrossEncoder(self.model_name, max_length=512)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load reranker model '{self.model_name}': {exc}"
            ) from exc

        logger.info("Reranker model '%s' loaded.", self.model_name)
        return self._model
