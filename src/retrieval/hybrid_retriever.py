"""
hybrid_retriever.py — Fuse dense (ChromaDB) + sparse (BM25) via Reciprocal Rank Fusion.

RRF formula (Cormack et al., 2009):
    score(chunk) = Σ  1 / (k + rank_i)
where k = 60 and rank_i is the 1-based rank of the chunk in result list i.

A chunk that appears in both lists receives contributions from both, naturally
up-weighting results that multiple retrieval methods agree are relevant.

Primary interface:
    retriever = HybridRetriever(collection_name, chunks_json_path)
    results = retriever.retrieve("what is multi-head attention?", top_k=20)
"""

import logging
from collections import defaultdict

from src.ingestion.embedder import Embedder
from src.ingestion.vectorstore import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

# Standard RRF constant from the original paper.  Larger k reduces the impact
# of very-high-rank documents; 60 is the canonical default.
_RRF_K = 60


class HybridRetriever:
    """Hybrid dense + sparse retriever using Reciprocal Rank Fusion.

    Args:
        collection_name: ChromaDB collection for this document.
        chunks_json_path: Path to the ``*_chunks.json`` BM25 cache file.
    """

    def __init__(self, collection_name: str, chunks_json_path: str) -> None:
        self.collection_name = collection_name

        # Instantiate once here and reuse — model loading is expensive.
        self.vectorstore = VectorStore()
        self.embedder = Embedder()
        self.bm25 = BM25Retriever.from_json(chunks_json_path)

        logger.info(
            "HybridRetriever ready: collection='%s', bm25_chunks=%d.",
            collection_name,
            len(self.bm25._chunks),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve and fuse dense and sparse results for a query.

        Args:
            query: User's natural-language question.
            top_k: Number of fused results to return.

        Returns:
            List of chunk dicts sorted by ``rrf_score`` descending.  Each dict
            contains the original chunk fields plus::

                {
                    "rrf_score":    float,       # fused RRF score
                    "in_dense":     bool,        # appeared in dense results
                    "in_sparse":    bool,        # appeared in sparse results
                    "dense_rank":   int | None,  # 1-based rank in dense list
                    "sparse_rank":  int | None,  # 1-based rank in sparse list
                }

        Raises:
            ValueError: If ``query`` is blank.
        """
        if not query.strip():
            raise ValueError("HybridRetriever.retrieve() received a blank query.")

        # ── Step 1: Dense retrieval ──────────────────────────────────────────
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vectorstore.query(
            query_embedding=query_embedding,
            collection_name=self.collection_name,
            top_k=top_k,
        )

        # ── Step 2: Sparse (BM25) retrieval ─────────────────────────────────
        sparse_results = self.bm25.retrieve(query, top_k=top_k)

        # ── Step 3: Reciprocal Rank Fusion ───────────────────────────────────
        fused = self._rrf_merge(dense_results, sparse_results, top_k)

        # ── Logging: fusion overlap stats ────────────────────────────────────
        dense_only = sum(1 for r in fused if r["in_dense"] and not r["in_sparse"])
        sparse_only = sum(1 for r in fused if r["in_sparse"] and not r["in_dense"])
        both = sum(1 for r in fused if r["in_dense"] and r["in_sparse"])
        logger.info(
            "RRF fusion: %d dense-only, %d sparse-only, %d in-both → %d returned.",
            dense_only,
            sparse_only,
            both,
            len(fused),
        )
        return fused

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Apply RRF to merge two ranked result lists into one.

        Args:
            dense_results: Ranked list from ChromaDB (best first).
            sparse_results: Ranked list from BM25 (best first).
            top_k: How many fused results to return.

        Returns:
            Fused, sorted list of chunk dicts (top_k or fewer).
        """
        # Accumulators keyed by chunk_id.
        rrf_scores: dict[str, float] = defaultdict(float)
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        chunk_data: dict[str, dict] = {}

        # Accumulate dense contributions.
        for rank_0, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rank = rank_0 + 1                          # convert to 1-based
            rrf_scores[cid] += 1.0 / (_RRF_K + rank)
            dense_ranks[cid] = rank
            if cid not in chunk_data:
                chunk_data[cid] = {
                    k: v for k, v in chunk.items()
                    if k not in ("similarity_score",)   # drop retriever-specific scores
                }

        # Accumulate sparse contributions.
        for rank_0, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rank = rank_0 + 1
            rrf_scores[cid] += 1.0 / (_RRF_K + rank)
            sparse_ranks[cid] = rank
            if cid not in chunk_data:
                chunk_data[cid] = {
                    k: v for k, v in chunk.items()
                    if k not in ("bm25_score",)
                }

        # Sort by fused RRF score and return top_k.
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # ── Table chunk injection ────────────────────────────────────────────
        # Table chunks score poorly in dense retrieval (bge-small-en-v1.5 cannot
        # represent pipe-separated markdown numbers).  A financial table that ranks
        # #2 in BM25 may still lose to prose chunks that rank in BOTH lists.
        # Fix: if a table chunk ranks in the BM25 top-5 but isn't already in the
        # RRF output, force-inject it so the cross-encoder can assess it.
        # The reranker already exempts table chunks from the relevance threshold.
        _TABLE_BM25_RANK_LIMIT = 5   # only inject if BM25 rank <= this
        _TABLE_INJECT_MAX = 3        # cap on extra injected chunks
        injected = 0
        result_id_set = set(sorted_ids)
        for rank_0, sparse_chunk in enumerate(sparse_results):
            if injected >= _TABLE_INJECT_MAX:
                break
            if rank_0 >= _TABLE_BM25_RANK_LIMIT:
                break
            cid = sparse_chunk["chunk_id"]
            if cid not in result_id_set and sparse_chunk.get("chunk_type") == "table":
                sorted_ids.append(cid)
                result_id_set.add(cid)
                injected += 1
                logger.info(
                    "Table injection: chunk_id=%s BM25_rank=%d (chunk_type=table).",
                    cid,
                    rank_0 + 1,
                )

        results = []
        for cid in sorted_ids:
            entry = dict(chunk_data[cid])          # copy so we don't mutate cache
            entry["rrf_score"] = rrf_scores[cid]
            entry["in_dense"] = cid in dense_ranks
            entry["in_sparse"] = cid in sparse_ranks
            entry["dense_rank"] = dense_ranks.get(cid)     # None if absent
            entry["sparse_rank"] = sparse_ranks.get(cid)   # None if absent
            results.append(entry)

        return results
