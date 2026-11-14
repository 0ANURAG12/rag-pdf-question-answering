"""
POST /api/ingest

Accepts a PDF file and runs it through the full indexing pipeline:
extract text -> chunk -> embed -> store in ChromaDB.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import settings
from app.exceptions import EmptyDocumentError
from app.logging_config import get_logger
from app.schemas import IngestResponse
from app.services import pdf_service, chunking_service, embedding_service, vectorstore_service

router = APIRouter()

logger = get_logger(__name__)

_PDF_MAGIC_BYTES = b"%PDF-"


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    # Validate that the uploaded file is actually a PDF before processing.
    if file.content_type not in ("application/pdf", "application/octet-stream") and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF documents are supported.")

    file_bytes = await file.read()

    if not file_bytes.startswith(_PDF_MAGIC_BYTES):
        raise HTTPException(status_code=400, detail="Invalid file. The uploaded file does not appear to be a valid PDF.")

    logger.info("ingest.start", extra={"file_name": file.filename, "file_size_bytes": len(file_bytes)})

    try:
        text = pdf_service.extract_text(file_bytes)
        chunks = chunking_service.split_text(text)
        embeddings = embedding_service.embed_texts(chunks)
        chunks_added = vectorstore_service.add_chunks(chunks, embeddings, source=file.filename)
    except EmptyDocumentError:
        logger.warning("ingest.empty_document", extra={"filename": file.filename})
        raise

    logger.info("ingest.success", extra={"filename": file.filename, "chunks_added": chunks_added})

    return IngestResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message="Document processed and indexed successfully.",
    )
