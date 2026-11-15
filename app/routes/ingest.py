"""
POST /api/ingest
GET  /api/ingest/{job_id}

Ingestion runs in the background so large PDFs do not hold the HTTP
request open. POST validates the upload, creates a job, kicks off the
extract -> chunk -> embed -> store pipeline, and returns a job_id the
client can poll with GET until the job reaches a terminal state.
"""

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.exceptions import EmptyDocumentError, InvalidPdfError
from app.logging_config import get_logger
from app.schemas import IngestJobResponse, IngestJobStatus
from app.services import ingestion_job_service, pdf_service

router = APIRouter()

logger = get_logger(__name__)

_PDF_MAGIC_BYTES = b"%PDF-"

# Tracks fire-and-forget ingestion tasks so the event loop never drops them.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


@router.post("/ingest", response_model=IngestJobResponse)
async def ingest_document(file: UploadFile = File(...)):
    # Validate that the uploaded file is actually a PDF before processing.
    if file.content_type not in ("application/pdf", "application/octet-stream") and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF documents are supported.")

    file_bytes = await file.read()

    if not file_bytes.startswith(_PDF_MAGIC_BYTES):
        raise HTTPException(status_code=400, detail="Invalid file. The uploaded file does not appear to be a valid PDF.")

    logger.info("ingest.start", extra={"file_name": file.filename, "file_size_bytes": len(file_bytes)})

    try:
        # Parse once, up front, so bad PDFs fail fast instead of creating a
        # job that is guaranteed to fail.
        text = await asyncio.to_thread(pdf_service.extract_text, file_bytes)
    except EmptyDocumentError:
        raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")
    except InvalidPdfError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF. It may be corrupted.",
        )

    job = ingestion_job_service.get_job_store().create(file.filename)
    task = asyncio.create_task(ingestion_job_service.run_ingestion(job, text))
    # Keep a reference so the task is not garbage-collected mid-run.
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return IngestJobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        message="Document accepted for indexing. Poll GET /api/ingest/{job_id} for progress.",
    )


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def get_ingest_status(job_id: str):
    job = ingestion_job_service.get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No ingestion job with id {job_id}")
    return IngestJobStatus(**job.to_dict())
