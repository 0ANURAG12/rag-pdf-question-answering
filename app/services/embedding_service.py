"""
Stage 3: Turn text into dense vector embeddings using Sentence Transformers.

The model is loaded once, when this module is first imported, and reused
for every request rather than being reloaded each time.
"""

from typing import List
from sentence_transformers import SentenceTransformer

from app.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of document chunks (used during ingestion)."""
    vectors = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single user question (used during retrieval)."""
    vector = _model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return vector[0].tolist()
