"""Citation extraction parses [Cite: source | Chunk: index] tags."""

from app.services import citation_service


def test_extract_citations_finds_tags():
    answer = "Alpha claim [Cite: a.pdf | Chunk: 0] and beta claim [Cite: b.pdf | Chunk: 2]."
    citations = citation_service.extract_citations(answer)
    assert citations == [
        {"source": "a.pdf", "chunk_index": 0},
        {"source": "b.pdf", "chunk_index": 2},
    ]


def test_extract_citations_returns_empty_when_absent():
    assert citation_service.extract_citations("no tags here") == []


def test_mark_cited_sources_flags_only_cited_chunks():
    chunks = [
        {"source": "a.pdf", "chunk_index": 0, "content": "x"},
        {"source": "a.pdf", "chunk_index": 1, "content": "y"},
    ]
    marked = citation_service.mark_cited_sources("[Cite: a.pdf | Chunk: 1]", chunks)
    assert marked[0]["cited"] is False
    assert marked[1]["cited"] is True
    # originals are not mutated
    assert "cited" not in chunks[0]


def test_filter_cited_keeps_only_cited():
    chunks = [
        {"source": "a.pdf", "chunk_index": 0, "cited": True},
        {"source": "a.pdf", "chunk_index": 1, "cited": False},
    ]
    assert citation_service.filter_cited(chunks) == [chunks[0]]


def test_citations_deduplicated_by_source_and_index():
    answer = "[Cite: a.pdf | Chunk: 0] again [Cite: a.pdf | Chunk: 0]"
    assert len(citation_service.extract_citations(answer)) == 2  # raw mentions
    # but mark_cited_sources collapses them by (source, chunk_index)
    marked = citation_service.mark_cited_sources(answer, [{"source": "a.pdf", "chunk_index": 0}])
    assert marked[0]["cited"] is True
