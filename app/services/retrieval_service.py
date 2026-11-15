"""
Stage 5: Retrieval.

Turns a user's question into embeddings and asks the vector store for the
most relevant chunks. When query expansion is enabled, a hypothetical
answer is also generated and used as a second query, so chunks that match
either phrasing are candidates. The result is then reranked by the
cross-encoder in rerank_service for higher precision before being returned.
"""

from typing import List, Dict, Any

from app.config import settings
from app.exceptions import NoDocumentsIndexedError
from app.logging_config import get_logger
from app.services import embedding_service, query_expansion_service, rerank_service, vectorstore_service

logger = get_logger(__name__)


async def _retrieve_candidates(question: str, fetch_k: int) -> List[Dict[str, Any]]:
    """Retrieve candidates for the question, merging results from all expanded queries."""
    queries = await query_expansion_service.expand_query(question)

    merged: Dict[str, Dict[str, Any]] = {}
    for query in queries:
        query_embedding = embedding_service.embed_query(query)
        results = vectorstore_service.query(query_embedding, top_k=fetch_k)
        for chunk in results:
            # Dedupe by (source, chunk_index); keep the best (lowest) distance.
            key = (chunk["source"], chunk["chunk_index"])
            existing = merged.get(key)
            if existing is None or chunk["distance"] < existing["distance"]:
                merged[key] = chunk

    # Sort by the best distance seen across all queries.
    candidates = sorted(merged.values(), key=lambda chunk: chunk["distance"])

    logger.info(
        "retrieval.candidates",
        extra={
            "queries": len(queries),
            "unique_candidates": len(candidates),
        },
    )
    return candidates


async def retrieve(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    if not vectorstore_service.has_documents():
        raise NoDocumentsIndexedError(
            "No documents have been uploaded yet. Please upload a PDF before asking questions."
        )

    k = settings.top_k if (top_k is None or top_k <= 0) else top_k
    k = min(k, settings.max_top_k)

    # Retrieve more candidates than we return so the reranker has room to
    # promote chunks the bi-encoder ranked too low.
    candidates = settings.rerank_candidates if settings.rerank_enabled else k
    fetch_k = max(candidates, k)

    chunks = await _retrieve_candidates(question, fetch_k)
    chunks = rerank_service.rerank(question, chunks, top_k=k)

    logger.info(
        "retrieval.retrieve",
        extra={
            "question_length": len(question),
            "top_k": k,
            "fetched": fetch_k,
            "chunks_returned": len(chunks),
            "reranked": settings.rerank_enabled,
        },
    )
    return chunks