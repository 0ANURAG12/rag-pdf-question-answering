"""
Stage 2: Split cleaned document text into smaller, overlapping chunks.

Two strategies are supported, selected by CHUNK_STRATEGY:

- "recursive" (default): LangChain's RecursiveCharacterTextSplitter, which
  breaks on paragraph boundaries first, then sentences, then words, with a
  fixed character overlap. Fast and predictable.
- "semantic": splits where the meaning of the document actually changes.
  Each sentence is embedded with the same model used for retrieval, and a
  new chunk starts when adjacent sentences become too dissimilar (cosine
  distance above SEMANTIC_THRESHOLD), still capped at chunk_size characters.
  This keeps semantically coherent content together instead of cutting it
  at arbitrary character positions.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.logging_config import get_logger
from app.services import embedding_service

logger = get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping the sentence-ending punctuation."""
    import re

    # Split on sentence enders followed by whitespace + a capital letter/digit,
    # which avoids breaking on common abbreviations.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance (1 - cosine similarity) between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _split_semantic(text: str) -> List[str]:
    """Split text into semantically coherent chunks, capped at chunk_size."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    sentence_embeddings = embedding_service.embed_texts(sentences)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for i, sentence in enumerate(sentences):
        # Start a new chunk if this sentence is semantically far from the
        # previous one, or if adding it would exceed the size cap.
        too_far = (
            i > 0
            and _cosine_distance(sentence_embeddings[i - 1], sentence_embeddings[i])
            > settings.semantic_threshold
        )
        too_big = current_len + len(sentence) > settings.chunk_size and current

        if too_far or too_big:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_text(text: str) -> List[str]:
    """Split a document's text into a list of chunk strings."""
    strategy = settings.chunk_strategy.lower()

    if strategy == "semantic":
        chunks = _split_semantic(text)
    else:
        chunks = _splitter.split_text(text)

    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    logger.info(
        "chunking.split_text",
        extra={
            "input_chars": len(text),
            "chunks_created": len(chunks),
            "strategy": strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )
    return chunks