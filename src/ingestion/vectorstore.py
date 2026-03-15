"""
vectorstore.py — ChromaDB-backed vector store with disk persistence.

Manages one ChromaDB collection per document. Provides idempotent ingestion
(chunk_id is used as the ChromaDB document ID) and structured result dicts.

Primary interface:
    vs = VectorStore()
    vs.ingest_chunks(chunks_with_embeddings, collection_name)
    results = vs.query(query_embedding, collection_name, top_k=5)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB collections for document retrieval.

    Each document is stored in a dedicated collection identified by a
    sanitised collection name derived from its filename.

    Args:
        persist_path: Directory where ChromaDB stores data on disk.
            Reads ``VECTORSTORE_PATH`` from the environment if not provided;
            falls back to ``./vectorstore``.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.persist_path: str = (
            persist_path
            or os.getenv("VECTORSTORE_PATH", "./vectorstore")
        )
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        self._client = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def ingest_chunks(self, chunks: list[dict], collection_name: str) -> int:
        """Upsert embedded chunks into a ChromaDB collection.

        Uses ``chunk_id`` as the ChromaDB document ID, making this call
        fully idempotent — running it twice on the same document is safe.

        Args:
            chunks: Chunk dicts that already contain an ``"embedding"`` key
                (output of :meth:`Embedder.embed_chunks`).
            collection_name: Target ChromaDB collection name.

        Returns:
            Number of chunks successfully stored.

        Raises:
            ValueError: If ``chunks`` is empty or any chunk is missing its
                ``"embedding"`` key.
        """
        if not chunks:
            raise ValueError("ingest_chunks received an empty chunk list.")
        if "embedding" not in chunks[0]:
            raise ValueError(
                "Chunks must have an 'embedding' key. "
                "Run Embedder.embed_chunks() before calling ingest_chunks()."
            )

        collection = self._get_collection(collection_name)

        ids = [c["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "page_number": c["page_number"],
                "source_filename": c["source_filename"],
                "chunk_id": c["chunk_id"],
                "chunk_index": c["chunk_index"],
                "chunk_type": c.get("chunk_type", "text"),
                # bbox is a list of 4 floats; ChromaDB metadata must be scalar,
                # so we JSON-encode it and decode on read.
                "bbox": json.dumps(c["bbox"]) if c.get("bbox") else "",
            }
            for c in chunks
        ]

        # Upsert in batches of 500 to avoid hitting ChromaDB's payload limits.
        batch_size = 500
        total = len(chunks)
        stored = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            stored += end - start
            logger.info(
                "Ingested %d/%d chunks into '%s'.", stored, total, collection_name
            )

        return stored

    def query(
        self,
        query_embedding: list[float],
        collection_name: str,
        top_k: int,
    ) -> list[dict]:
        """Retrieve the most similar chunks for a query embedding.

        Args:
            query_embedding: Dense query vector (from :meth:`Embedder.embed_query`).
            collection_name: ChromaDB collection to search.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts sorted by ``similarity_score`` descending::

                {
                    "text":             str,
                    "page_number":      int,
                    "source_filename":  str,
                    "chunk_id":         str,
                    "chunk_index":      int,
                    "similarity_score": float,  # cosine similarity [0, 1]
                }

        Raises:
            ValueError: If the collection does not exist or is empty.
        """
        collection = self._get_collection(collection_name)
        count = collection.count()
        if count == 0:
            raise ValueError(
                f"Collection '{collection_name}' is empty. Ingest the document first."
            )

        n_results = min(top_k, count)
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB returns cosine *distance* (lower = more similar).
            # Convert to similarity score in [0, 1].
            similarity = float(1.0 - dist)
            bbox_raw = meta.get("bbox", "")
            results.append(
                {
                    "text": doc,
                    "page_number": meta["page_number"],
                    "source_filename": meta["source_filename"],
                    "chunk_id": meta["chunk_id"],
                    "chunk_index": meta["chunk_index"],
                    "chunk_type": meta.get("chunk_type", "text"),
                    "bbox": json.loads(bbox_raw) if bbox_raw else None,
                    "similarity_score": similarity,
                }
            )

        # Sort by similarity descending (ChromaDB usually already does this,
        # but we guarantee it here).
        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results

    def collection_exists(self, collection_name: str) -> bool:
        """Return True if the named collection already has stored documents.

        Used by the ingestion pipeline to skip re-embedding a document that
        has already been processed.

        Args:
            collection_name: Collection to check.

        Returns:
            ``True`` if the collection exists and contains at least one document.
        """
        try:
            client = self._get_client()
            existing = [c.name for c in client.list_collections()]
            if collection_name not in existing:
                return False
            collection = client.get_collection(collection_name)
            return collection.count() > 0
        except Exception:
            return False

    def get_collection_stats(self, collection_name: str) -> dict:
        """Return basic statistics about a stored collection.

        Args:
            collection_name: Target collection.

        Returns:
            Dict with keys: ``collection_name``, ``document_count``,
            ``persist_path``.
        """
        try:
            collection = self._get_collection(collection_name)
            count = collection.count()
        except Exception:
            count = 0

        return {
            "collection_name": collection_name,
            "document_count": count,
            "persist_path": str(Path(self.persist_path).resolve()),
        }

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its stored data.

        Args:
            collection_name: Collection to delete.
        """
        client = self._get_client()
        client.delete_collection(collection_name)
        logger.info("Deleted collection '%s'.", collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return a shared ChromaDB persistent client (lazy init).

        Returns:
            A :class:`chromadb.PersistentClient` instance.
        """
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.debug("ChromaDB client opened at '%s'.", self.persist_path)
        return self._client

    def _get_collection(self, collection_name: str):
        """Open or create a named ChromaDB collection using cosine distance.

        Args:
            collection_name: Name of the collection.

        Returns:
            A :class:`chromadb.Collection` instance.
        """
        client = self._get_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection
