"""End-to-end query flow through the FastAPI TestClient, fully mocked."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import fake_embed


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client, store, monkeypatch):
    import app.services.vectorstore_service as vs

    monkeypatch.setattr(vs, "_default_store", store)
    store.add_chunks(
        ["The sky is blue.", "Grass is green."],
        [fake_embed("The sky is blue."), fake_embed("Grass is green.")],
        source="nature.pdf",
    )

    resp = client.post("/api/query", json={"question": "What color is the sky?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "mock answer"
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"] == "nature.pdf"


def test_query_rejects_empty_question(client):
    assert client.post("/api/query", json={"question": ""}).status_code == 422


def test_query_rejects_overlong_question(client):
    assert client.post("/api/query", json={"question": "x" * 2001}).status_code == 422


def test_query_rejects_bad_top_k(client):
    assert client.post("/api/query", json={"question": "hi", "top_k": 99}).status_code == 422


def test_ingest_rejects_non_pdf(client):
    resp = client.post("/api/ingest", files={"file": ("x.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_ingest_rejects_bad_magic_bytes(client):
    resp = client.post("/api/ingest", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert resp.status_code == 400
