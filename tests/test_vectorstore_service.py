"""VectorStore round-trip: add, query, count, and the injectable client."""

from tests.conftest import fake_embed

from app.services.vectorstore_service import VectorStore


def test_add_and_query_roundtrip(store):
    chunks = ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma"]
    embeddings = fake_embed(chunks)
    added = store.add_chunks(chunks, embeddings, source="doc.pdf")
    assert added == 3

    results = store.query(fake_embed("alpha beta gamma"), top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "doc.pdf"
    assert "content" in results[0] and "chunk_index" in results[0]


def test_add_clears_previous_documents(store):
    store.add_chunks(["first"], fake_embed(["first"]), source="a.pdf")
    store.add_chunks(["second"], fake_embed(["second"]), source="b.pdf")
    assert store.count_documents() == 1
    assert store.query(fake_embed("second"), top_k=5)[0]["source"] == "b.pdf"


def test_query_on_empty_store(store):
    assert store.query(fake_embed("anything"), top_k=3) == []
    assert store.has_documents() is False


def test_count_and_has_documents(store):
    assert store.count_documents() == 0
    store.add_chunks(["x"], fake_embed(["x"]), source="c.pdf")
    assert store.count_documents() == 1
    assert store.has_documents() is True
