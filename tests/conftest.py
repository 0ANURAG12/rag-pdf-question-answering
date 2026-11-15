"""Shared fixtures: a vector store fed fake embeddings, no model downloads."""

import pytest

import app.services.embedding_service as embedding_service
import app.services.llm_service as llm_service
import app.services.rerank_service as rerank_service
from app.services.vectorstore_service import VectorStore


class FakeEmbeddingModel:
    """Deterministic 32-dim hash-based embeddings."""

    def encode(self, texts, **kwargs):
        import hashlib

        vectors = []
        for text in texts:
            digest = hashlib.md5(str(text).encode()).digest() * 2
            vectors.append([b / 255.0 for b in digest])
        return vectors


def fake_embed(texts):
    """
    Fake embedding matching the real service's shapes:
    a bare string gets a flat vector (embed_query), a list gets a batch
    of vectors (embed_texts).
    """
    model = FakeEmbeddingModel()
    if isinstance(texts, str):
        return model.encode([texts])[0]
    return model.encode(texts)


async def _fake_generate_answer(prompt: str) -> str:
    return "mock answer"


class _FakeCrossEncoder:
    """Deterministic stand-in: score by question/content word overlap."""

    def predict(self, pairs):
        # Each pair is (question_text, chunk_content_text).
        scores = []
        for question, content in pairs:
            q_words = set(question.lower().split())
            c_words = set(content.lower().split())
            scores.append(len(q_words & c_words))
        return scores


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch):
    """Never hit the network for embeddings, reranking, or the LLM."""
    monkeypatch.setattr(embedding_service, "embed_texts", fake_embed)
    monkeypatch.setattr(embedding_service, "embed_query", fake_embed)
    monkeypatch.setattr(llm_service, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(llm_service, "list_models", lambda: [{"name": "llama3.2:latest"}])
    monkeypatch.setattr(rerank_service, "_get_model", lambda: _FakeCrossEncoder())


@pytest.fixture()
def store():
    """A VectorStore backed by a fresh in-memory ChromaDB client per test."""
    import uuid

    import chromadb

    # A unique collection name per test: EphemeralClient is a singleton, so
    # reusing the default collection would leak documents between tests.
    return VectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"test-{uuid.uuid4().hex[:8]}",
    )
