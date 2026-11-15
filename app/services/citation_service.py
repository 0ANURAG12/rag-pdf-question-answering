"""
Citation extraction.

The prompt tells the LLM to cite the chunks it used with the tag
[Cite: <source> | Chunk: <index>]. This module scans the answer for those
tags and maps them back to the retrieved chunks, so the response can
separate the sources that actually contributed from the ones that were
retrieved but not used.
"""

import re
from typing import List, Dict, Any, Optional

from app.logging_config import get_logger

logger = get_logger(__name__)

_CITE_PATTERN = re.compile(r"\[Cite:\s*(?P<source>[^|\]]+)\s*\|\s*Chunk:\s*(?P<index>\d+)\s*\]")


def extract_citations(answer: str) -> List[Dict[str, Any]]:
    """Return the list of (source, chunk_index) pairs cited in the answer."""
    return [
        {"source": match.group("source").strip(), "chunk_index": int(match.group("index"))}
        for match in _CITE_PATTERN.finditer(answer)
    ]


def mark_cited_sources(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the chunks with a `cited` flag: True when the answer cites that
    (source, chunk_index) pair, False otherwise. Chunks are otherwise returned
    unchanged, preserving order.
    """
    cited_keys = {(c["source"], c["chunk_index"]) for c in extract_citations(answer)}

    marked = []
    for chunk in chunks:
        entry = dict(chunk)
        entry["cited"] = (chunk["source"], chunk["chunk_index"]) in cited_keys
        marked.append(entry)

    cited_count = sum(1 for c in marked if c["cited"])
    logger.info(
        "citation.extract",
        extra={
            "citations_found": len(cited_keys),
            "chunks_cited": cited_count,
            "chunks_total": len(marked),
        },
    )
    return marked


def filter_cited(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the chunks that were cited in the answer."""
    return [chunk for chunk in chunks if chunk.get("cited")]
