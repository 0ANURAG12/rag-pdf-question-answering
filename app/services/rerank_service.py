"""
Reranking stage (between Stage 5 retrieval and Stage 6 prompt building).

The bi-encoder used for retrieval (all-MiniLM-L6-v2) encodes the question
and each chunk independently, so it can miss chunks that only match the
question in combination with other words. A cross-encoder instead scores
each (question, chunk) pair jointly, which is far more accurate — but far
slower, which is why we only run it on the small set of candidates
retrieved in the first pass.

Flow: retrieve RERANK_CANDIDATES chunks -> rerank -> keep top_k.
"""

from typing import List, Dict, Any

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_model = None


def _get_model():
    """Lazy-initialize the cross-encoder once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("rerank.loading_model", extra={"model": settings.rerank_model})
        _model = CrossEncoder(settings.rerank_model)
    return _model


def rerank(question: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Score each chunk against the question with the cross-encoder and return
    the top_k chunks sorted by relevance. Falls back to the input order if
    the reranker fails, so retrieval never breaks because of reranking.
    """
    if not settings.rerank_enabled or not chunks:
        return chunks

    try:
        model = _get_model()
        pairs = [(question, chunk["content"]) for chunk in chunks]
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001 — never let reranking break retrieval
        logger.warning("rerank.failed_using_initial_order", extra={"error": str(exc)})
        return chunks

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    logger.info(
        "rerank.reranked",
        extra={
            "candidates": len(chunks),
            "kept": min(top_k, len(chunks)),
            "best_score": float(scored[0][1]),
            "worst_score": float(scored[-1][1]),
        },
    )
    return [chunk for chunk, _score in scored[:top_k]]
