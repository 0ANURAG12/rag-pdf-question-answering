"""
Background ingestion job registry.

Each PDF upload creates a job record the client can poll with
GET /api/ingest/{job_id} until it reaches a terminal state
(completed or failed), so large documents no longer hold the
HTTP request open for the whole extract -> chunk -> embed -> store
pipeline.

Records live in memory: this is a single-process prototype, so jobs
are lost if the service restarts.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

logger = get_logger(__name__)

PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"

_TERMINAL_STATES = {COMPLETED, FAILED}


class IngestionJob:
    """State for one background ingestion run."""

    def __init__(self, filename: str) -> None:
        self.job_id = str(uuid.uuid4())
        self.filename = filename
        self.status = PENDING
        self.chunks_added: Optional[int] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = self.created_at

    def mark_processing(self) -> None:
        self.status = PROCESSING
        self.updated_at = time.time()

    def mark_completed(self, chunks_added: int) -> None:
        self.status = COMPLETED
        self.chunks_added = chunks_added
        self.updated_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = FAILED
        self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "chunks_added": self.chunks_added,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class IngestionJobStore:
    """Keeps IngestionJob records keyed by job_id."""

    def __init__(self) -> None:
        self._jobs: Dict[str, IngestionJob] = {}

    def create(self, filename: str) -> IngestionJob:
        job = IngestionJob(filename)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[IngestionJob]:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent jobs first."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.to_dict() for job in jobs[:limit]]


# Shared registry used by the ingestion route.
_job_store = IngestionJobStore()


def get_job_store() -> IngestionJobStore:
    return _job_store


async def run_ingestion(job: IngestionJob, text: str) -> None:
    """
    Run the chunk -> embed -> store pipeline for `job`, updating it in place.

    The PDF has already been parsed by the caller (which fails fast on
    invalid input); the blocking steps here are offloaded to a thread so
    they never block the event loop.
    """
    from app.services import chunking_service, embedding_service, vectorstore_service

    job.mark_processing()
    logger.info("ingest.job.started", extra={"job_id": job.job_id, "file_name": job.filename})

    try:
        chunks = await asyncio.to_thread(chunking_service.split_text, text)
        embeddings = await asyncio.to_thread(embedding_service.embed_texts, chunks)
        chunks_added = await asyncio.to_thread(
            vectorstore_service.add_chunks, chunks, embeddings, source=job.filename
        )
    except Exception as exc:  # noqa: BLE001 — recorded on the job, not raised
        job.mark_failed(str(exc))
        logger.error(
            "ingest.job.failed",
            extra={"job_id": job.job_id, "file_name": job.filename, "error": str(exc)},
        )
        return

    job.mark_completed(chunks_added)
    logger.info(
        "ingest.job.completed",
        extra={"job_id": job.job_id, "file_name": job.filename, "chunks_added": chunks_added},
    )


def is_terminal(status: str) -> bool:
    return status in _TERMINAL_STATES
