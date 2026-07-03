"""
Pydantic schemas (DTOs) used for request and response bodies.
Kept in a single file since the service is small and this is
easier to scan than hunting across several schema files.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
