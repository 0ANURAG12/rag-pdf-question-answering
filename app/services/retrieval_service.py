"""
Stage 5: Retrieval.

Turns a user's question into an embedding and asks the vector store
for the most relevant chunks.
"""

from typing import List, Dict, Any

from app.config import settings
from app.exceptions import NoDocumentsIndexedError
from app.logging_config import get_logger
from app.services import embedding_service, vectorstore_service

logger = get_logger(__name__)


def retrieve(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    if not vectorstore_service.has_documents():
        raise NoDocumentsIndexedError(
            "No documents have been uploaded yet. Please upload a PDF before asking questions."
        )

    k = settings.top_k if (top_k is None or top_k <= 0) else top_k
    query_embedding = embedding_service.embed_query(question)
    chunks = vectorstore_service.query(query_embedding, top_k=k)

    logger.info(
        "retrieval.retrieve",
        extra={
            "question_length": len(question),
            "top_k": k,
            "chunks_returned": len(chunks),
        },
    )
    return chunks