"""PDF validation: magic bytes and empty-text failures."""

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from app.exceptions import EmptyDocumentError, InvalidPdfError
from app.services import pdf_service


def _make_pdf(text: str) -> bytes:
    """Build a valid single-page PDF carrying `text` via reportlab."""
    from io import BytesIO

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def test_extract_text_reads_pdf():
    text = pdf_service.extract_text(_make_pdf("Hello RAG"))
    assert "Hello RAG" in text


def test_extract_text_rejects_non_pdf():
    with pytest.raises(InvalidPdfError):
        pdf_service.extract_text(b"this is not a pdf at all")


def test_extract_text_raises_on_empty_document():
    with pytest.raises(EmptyDocumentError):
        pdf_service.extract_text(_make_pdf(""))
