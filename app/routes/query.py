"""
POST /api/query

Accepts a question and runs it through the full retrieval-generation
pipeline: retrieve relevant chunks -> build prompt -> ask Ollama ->
extract citations -> return the answer with the chunks that supported it.
"""

from fastapi import APIRouter

from app.config import settings
from app.logging_config import get_logger
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import retrieval_service, prompt_service, llm_service, citation_service

router = APIRouter()

logger = get_logger(__name__)

_NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."


@router.post("/query", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    logger.info(
        "query.start",
        extra={
            "question_length": len(payload.question),
            "top_k": payload.top_k or settings.top_k,
        },
    )

    chunks = await retrieval_service.retrieve(payload.question, top_k=payload.top_k)
    prompt = prompt_service.build_prompt(payload.question, chunks)
    answer = await llm_service.generate_answer(prompt)

    # If LLM indicates no relevant information was found, return empty sources
    if answer.strip() == _NOT_FOUND_MESSAGE:
        sources = []
        citations = []
    else:
        marked = citation_service.mark_cited_sources(answer, chunks)
        sources = [
            SourceChunk(
                content=chunk["content"],
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
                distance=chunk.get("distance"),
                cited=chunk.get("cited"),
            )
            for chunk in marked
        ]
        # Citations: only the sources that actually contributed to the answer.
        citations = [
            SourceChunk(
                content=chunk["content"],
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
                distance=chunk.get("distance"),
                cited=True,
            )
            for chunk in citation_service.filter_cited(marked)
        ]

    logger.info(
        "query.end",
        extra={
            "answer_length": len(answer),
            "sources_returned": len(sources),
            "citations_returned": len(citations),
        },
    )

    return QueryResponse(answer=answer, sources=sources, citations=citations)
