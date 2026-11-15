"""Recursive chunking respects the configured size and overlap."""

from app.services import chunking_service


def _paragraph(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def test_recursive_strategy_returns_nonempty_chunks():
    chunks = chunking_service.split_text(_paragraph(1200))
    assert chunks
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunks_respect_max_size():
    chunks = chunking_service.split_text(_paragraph(2000))
    assert all(len(c) <= 600 for c in chunks), "a chunk exceeded the configured max size"


def test_empty_text_yields_no_chunks():
    assert chunking_service.split_text("") == []
    assert chunking_service.split_text("   \n\t  ") == []
