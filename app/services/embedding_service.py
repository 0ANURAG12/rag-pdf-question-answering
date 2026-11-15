"""
Stage 3: Turn text into dense vector embeddings using Sentence Transformers.

The model is loaded lazily on first use, so /health and startup don't pay
the initialization cost unless embeddings are actually needed.
"""

import threading
from typing import List
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Lazy-initialize and return the shared SentenceTransformer instance."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("embedding.loading_model", extra={"model": settings.embedding_model})
                _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of document chunks (used during ingestion)."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    logger.info(
        "embedding.embed_texts",
        extra={"texts": len(texts), "dimensions": vectors.shape[1]},
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single user question (used during retrieval)."""
    model = _get_model()
    vector = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return vector[0].tolist()