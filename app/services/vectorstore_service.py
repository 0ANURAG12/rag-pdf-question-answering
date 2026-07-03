"""
Stage 4: Store and search chunk embeddings using ChromaDB.

A raw chromadb.PersistentClient is used directly, rather than going
through a framework wrapper, so the collection calls stay easy to
read and easy to explain.
"""

import uuid
from typing import List, Dict, Any

import chromadb

from app.config import settings

_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection(name=settings.collection_name)


def add_chunks(chunks: List[str], embeddings: List[List[float]], source: str) -> int:
    """Store chunks and their embeddings, tagged with the source filename."""
    # Clear all existing vectors before indexing the new document.
    # This service is a single-document RAG prototype — only one PDF is active at a time.
    existing = _collection.get()
    if existing["ids"]:
        _collection.delete(ids=existing["ids"])

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Return the top_k most similar chunks to the given query embedding."""
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "content": documents[i],
            "source": metadatas[i]["source"],
            "chunk_index": metadatas[i]["chunk_index"],
            "distance": distances[i],
        }
        for i in range(len(documents))
    ]


def has_documents() -> bool:
    return _collection.count() > 0
