"""
Stage 4: Store and search chunk embeddings using ChromaDB.

A raw chromadb.PersistentClient is used directly, rather than going
through a framework wrapper, so the collection calls stay easy to
read and easy to explain.

The VectorStore class accepts an injected chromadb client, so tests can
swap in an in-memory or mock client. The module-level functions delegate
to a shared default instance, keeping the existing call sites unchanged.
"""

import uuid
from typing import List, Dict, Any, Optional

import chromadb

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Thin wrapper over a ChromaDB collection, injectable for tests."""

    def __init__(
        self,
        client: Optional[chromadb.ClientAPI] = None,
        collection_name: Optional[str] = None,
    ):
        self._client = client or chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.collection_name
        )

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], source: str) -> int:
        """Store chunks and their embeddings, tagged with the source filename."""
        # Clear all existing vectors before indexing the new document.
        # This service is a single-document RAG prototype — only one PDF is active at a time.
        existing = self._collection.get()
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(
            "vectorstore.add_chunks",
            extra={
                "chunks_added": len(chunks),
                "source": source,
                "collection": settings.collection_name,
            },
        )
        return len(chunks)

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Return the top_k most similar chunks to the given query embedding."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        logger.info(
            "vectorstore.query",
            extra={
                "top_k": top_k,
                "returned": len(documents),
                "collection": settings.collection_name,
            },
        )

        return [
            {
                "content": documents[i],
                "source": metadatas[i]["source"],
                "chunk_index": metadatas[i]["chunk_index"],
                "distance": distances[i],
            }
            for i in range(len(documents))
        ]

    def has_documents(self) -> bool:
        return self._collection.count() > 0

    def count_documents(self) -> int:
        """Number of chunks currently indexed (used by /health/deep)."""
        return self._collection.count()


# Shared default instance used by the module-level functions below.
_default_store = VectorStore()


def add_chunks(chunks: List[str], embeddings: List[List[float]], source: str) -> int:
    return _default_store.add_chunks(chunks, embeddings, source)


def query(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    return _default_store.query(query_embedding, top_k=top_k)


def has_documents() -> bool:
    return _default_store.has_documents()


def count_documents() -> int:
    return _default_store.count_documents()
