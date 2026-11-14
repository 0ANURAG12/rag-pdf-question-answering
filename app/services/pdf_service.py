"""
Stage 1: Extract raw text from an uploaded PDF using PyPDF.
"""

import re
from io import BytesIO
from pypdf import PdfReader

from app.exceptions import EmptyDocumentError, InvalidPdfError
from app.logging_config import get_logger

logger = get_logger(__name__)


def _clean_page_text(text: str) -> str:
    """
    Normalize raw PyPDF page text:
    - Merge lines that are soft-wrapped
    - Preserve paragraph breaks (blank lines)
    - Remove repeated whitespace
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Preserve paragraph breaks (2+ newlines) with a placeholder
    text = re.sub(r"\n{2,}", "\x00PARAGRAPH\x00", text)
    
    # 3. Merge single newlines into spaces
    text = text.replace("\n", " ")
    
    # 4. Restore paragraph breaks
    text = text.replace("\x00PARAGRAPH\x00", "\n\n")
    
    # 5. Collapse repeated whitespace
    text = re.sub(r" {2,}", " ", text)
    
    # 6. Trim each paragraph
    paragraphs = [p.strip() for p in text.split("\n\n")]
    text = "\n\n".join(p for p in paragraphs if p)
    
    return text


def extract_text(file_bytes: bytes) -> str:
    """Read every page of a PDF and join the extracted text together."""
    try:
        reader = PdfReader(BytesIO(file_bytes))

        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(_clean_page_text(text))
    except Exception as exc:  # noqa: BLE001 — PyPDF raises several types
        logger.warning("pdf.invalid", extra={"error": str(exc)})
        raise InvalidPdfError(
            "The uploaded file could not be read as a PDF. It may be corrupted or not a real PDF."
        ) from exc

    full_text = "\n\n".join(pages_text).strip()

    if not full_text:
        logger.warning(
            "pdf.empty_document",
            extra={"pages": len(reader.pages)},
        )
        raise EmptyDocumentError(
            "No extractable text was found in this PDF. "
            "It may be a scanned image without a text layer."
        )

    logger.info(
        "pdf.extract_text",
        extra={"pages": len(reader.pages), "chars_extracted": len(full_text)},
    )
    return full_text
