"""
Pydantic schemas (DTOs) used for request and response bodies.
Kept in a single file since the service is small and this is
easier to scan than hunting across several schema files.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Deprecated synchronous response kept for reference; ingestion is now async."""

    filename: str
    chunks_added: int
    message: str


class IngestJobResponse(BaseModel):
    """Returned by POST /api/ingest — the upload was accepted as a background job."""

    job_id: str
    filename: str
    status: str
    message: str


class IngestJobStatus(BaseModel):
    """Returned by GET /api/ingest/{job_id} — current state of the job."""

    job_id: str
    filename: str
    status: str
    chunks_added: Optional[int] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_index: int
    distance: Optional[float] = None
    cited: Optional[bool] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    citations: Optional[List[SourceChunk]] = None
