"""Reranking behaviour: relevant chunks rise to the top; failures fall back."""

from app.services import rerank_service


def _chunks():
    return [
        {"content": "The mitochondria is the powerhouse of the cell."},
        {"content": "Photosynthesis converts sunlight into chemical energy."},
        {"content": "The mitochondria produces ATP through cellular respiration."},
    ]


def test_rerank_promotes_relevant_chunk():
    ranked = rerank_service.rerank("mitochondria function", _chunks(), top_k=3)
    assert len(ranked) == 3
    # Both mitochondria chunks should outrank the photosynthesis one.
    texts = [c["content"] for c in ranked]
    assert texts[0].startswith("The mitochondria")


def test_rerank_respects_top_k():
    ranked = rerank_service.rerank("energy", _chunks(), top_k=2)
    assert len(ranked) == 2


def test_rerank_handles_empty_candidates():
    assert rerank_service.rerank("anything", [], top_k=3) == []


def test_rerank_falls_back_on_model_failure(monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(rerank_service, "_get_model", _explode)
    # 4 chunks with top_k=3 so the rerank path actually runs
    chunks = _chunks() + [{"content": "Ribosomes synthesize proteins."}]
    ranked = rerank_service.rerank("mitochondria", chunks, top_k=3)
    assert ranked == chunks, "failed rerank must fall back to original order"
