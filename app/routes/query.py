"""
POST /api/query

Accepts a question and runs it through the full retrieval-generation
pipeline: retrieve relevant chunks -> build prompt -> ask Ollama ->
return the answer together with the chunks that supported it.
"""

from fastapi import APIRouter

from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import retrieval_service, prompt_service, llm_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    chunks = retrieval_service.retrieve(payload.question, top_k=payload.top_k)
    prompt = prompt_service.build_prompt(payload.question, chunks)
    answer = llm_service.generate_answer(prompt)

    # If LLM indicates no relevant information was found, return empty sources
    if answer.strip() == "I could not find this information in the uploaded documents.":
        sources = []
    else:
        sources = [
            SourceChunk(
                content=chunk["content"],
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in chunks
        ]

    return QueryResponse(answer=answer, sources=sources)
