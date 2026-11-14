"""
FastAPI application entrypoint for the RAG AI Service.

This service owns the entire AI pipeline described in the project
report: PDF parsing, chunking, embeddings, ChromaDB storage,
retrieval, prompt construction, and LLM generation via Ollama.
"""

import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import register_exception_handlers
from app.logging_config import setup_logging, get_logger
from app.routes import ingest, query
from app.services import llm_service, vectorstore_service

setup_logging(log_level=settings.log_level, json_logs=settings.json_logs)
logger = get_logger("ai-service")

app = FastAPI(
    title="RAG AI Service",
    description="Handles PDF ingestion and retrieval-augmented question answering.",
    version="1.1.0",
)

# Allow the Spring Boot backend (and browser during local testing) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info("request.start", extra={"request_id": request_id, "path": request.url.path})
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request.end",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "status_code": response.status_code,
        },
    )
    return response


register_exception_handlers(app)

app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.on_event("shutdown")
async def shutdown_event():
    """Close the shared HTTP client used to talk to Ollama."""
    await llm_service.close_client()


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


@app.get("/health/deep", tags=["Health"])
async def deep_health_check():
    """Check that every dependency the service relies on is reachable."""
    results = {"fastapi": "ok", "ollama": "ok", "chromadb": "ok"}
    detail = {}

    # Ollama — list available models as a connectivity probe.
    try:
        models = await llm_service.list_models()
        detail["ollama"] = {"models": [m.get("name") for m in models]}
    except Exception as exc:  # noqa: BLE001 — report any failure as unhealthy
        results["ollama"] = "unhealthy"
        detail["ollama"] = str(exc)

    # ChromaDB — count collections as a connectivity probe.
    try:
        count = vectorstore_service.count_documents()
        detail["chromadb"] = {"document_count": count}
    except Exception as exc:  # noqa: BLE001
        results["chromadb"] = "unhealthy"
        detail["chromadb"] = str(exc)

    unhealthy = {k for k, v in results.items() if v != "ok"}
    status_code = 503 if unhealthy else 200
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if not unhealthy else "degraded", "dependencies": detail},
    )
