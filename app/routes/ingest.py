"""
POST /api/ingest

Accepts a PDF file and runs it through the full indexing pipeline:
extract text -> chunk -> embed -> store in ChromaDB.
"""

from fastapi import APIRouter, UploadFile, File

from app.schemas import IngestResponse
from app.services import pdf_service, chunking_service, embedding_service, vectorstore_service

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    file_bytes = await file.read()

    text = pdf_service.extract_text(file_bytes)
    chunks = chunking_service.split_text(text)
    embeddings = embedding_service.embed_texts(chunks)
    chunks_added = vectorstore_service.add_chunks(chunks, embeddings, source=file.filename)

    return IngestResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message="Document processed and indexed successfully.",
    )
