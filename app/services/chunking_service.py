"""
Stage 2: Split cleaned document text into smaller, overlapping chunks
using LangChain's RecursiveCharacterTextSplitter.

The splitter tries to break text on paragraph boundaries first, then
sentences, then words, only falling back to a hard character cut if
nothing else fits within chunk_size. This keeps chunks semantically
coherent, as discussed in the project report.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_text(text: str) -> List[str]:
    """Split a document's text into a list of chunk strings."""
    chunks = _splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
