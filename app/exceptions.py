"""
Custom exceptions for the AI service, along with the FastAPI
exception handlers that turn them into clean JSON error responses.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class EmptyDocumentError(Exception):
    """Raised when a PDF contains no extractable text."""


class InvalidPdfError(Exception):
    """Raised when the uploaded file is not a valid or readable PDF."""


class NoDocumentsIndexedError(Exception):
    """Raised when a query is made before any document has been ingested."""


class OllamaServiceError(Exception):
    """Raised when the local Ollama server cannot be reached or errors out."""


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(EmptyDocumentError)
    async def empty_document_handler(request: Request, exc: EmptyDocumentError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(InvalidPdfError)
    async def invalid_pdf_handler(request: Request, exc: InvalidPdfError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(NoDocumentsIndexedError)
    async def no_documents_handler(request: Request, exc: NoDocumentsIndexedError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OllamaServiceError)
    async def ollama_error_handler(request: Request, exc: OllamaServiceError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected server error: {exc}"},
        )
