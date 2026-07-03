"""
FastAPI application entrypoint for the RAG AI Service.

This service owns the entire AI pipeline described in the project
report: PDF parsing, chunking, embeddings, ChromaDB storage,
retrieval, prompt construction, and LLM generation via Ollama.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.exceptions import register_exception_handlers
from app.routes import ingest, query

app = FastAPI(
    title="RAG AI Service",
    description="Handles PDF ingestion and retrieval-augmented question answering.",
    version="1.0.0",
)

# Allow the Spring Boot backend (and browser during local testing) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
