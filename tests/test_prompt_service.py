"""Prompt construction keeps the question, context, and citation tags intact."""

from app.services import prompt_service


def _chunks():
    return [
        {"source": "a.pdf", "chunk_index": 0, "content": "Alpha text."},
        {"source": "b.pdf", "chunk_index": 1, "content": "Beta text."},
    ]


def test_prompt_contains_question_and_context():
    prompt = prompt_service.build_prompt("What is alpha?", _chunks())
    assert "What is alpha?" in prompt
    assert "Alpha text." in prompt
    assert "Beta text." in prompt
    assert "a.pdf" in prompt and "Chunk: 0" in prompt
    assert "b.pdf" in prompt and "Chunk: 1" in prompt


def test_prompt_demands_citations():
    prompt = prompt_service.build_prompt("q", _chunks())
    assert "[Cite:" in prompt, "prompt must instruct the model to emit citation tags"


def test_prompt_with_no_chunks_still_answers_from_context():
    prompt = prompt_service.build_prompt("anything", [])
    assert "anything" in prompt
    assert "Context:" in prompt


def test_prompt_tolerates_braces_in_content():
    prompt = prompt_service.build_prompt("q", [{"source": "x.pdf", "chunk_index": 0, "content": "{braces}"}])
    assert "{braces}" in prompt
