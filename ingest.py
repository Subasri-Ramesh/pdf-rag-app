"""PDF validation, text extraction, and page-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pymupdf as fitz


MAX_PDF_BYTES = 25 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 200


class PDFIngestionError(ValueError):
    """Base class for user-correctable PDF ingestion errors."""


class ScannedPDFError(PDFIngestionError):
    """Raised when a PDF has too little extractable text and likely needs OCR."""


@dataclass(frozen=True)
class PageText:
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    filename: str
    page_count: int
    chunks: tuple[TextChunk, ...]


def safe_filename(filename: str) -> str:
    """Return a basename safe to display and store as metadata."""
    cleaned = Path(filename or "").name.strip().replace("\x00", "")
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    if not cleaned or not cleaned.lower().endswith(".pdf"):
        raise PDFIngestionError("The uploaded file must have a .pdf extension.")
    return cleaned[:180]


def validate_pdf(filename: str, data: bytes, max_bytes: int = MAX_PDF_BYTES) -> str:
    """Validate the upload and return its sanitized filename."""
    cleaned = safe_filename(filename)
    if not data:
        raise PDFIngestionError("The uploaded PDF is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise PDFIngestionError(f"The PDF exceeds the {limit_mb} MB upload limit.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise PDFIngestionError("The file does not have a valid PDF signature.")
    return cleaned


def extract_pages(filename: str, data: bytes) -> list[PageText]:
    """Extract text from every page, retaining one-based page numbers."""
    cleaned = validate_pdf(filename, data)
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFIngestionError("The PDF is corrupted or cannot be opened.") from exc

    try:
        if document.needs_pass:
            raise PDFIngestionError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise PDFIngestionError("The PDF contains no pages.")

        pages: list[PageText] = []
        for index, page in enumerate(document):
            text = re.sub(r"[ \t]+", " ", page.get_text("text"))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages.append(PageText(cleaned, index + 1, text))
    finally:
        document.close()

    meaningful_pages = sum(len(page.text) >= 40 for page in pages)
    total_chars = sum(len(page.text) for page in pages)
    if total_chars < max(80, len(pages) * 20) or meaningful_pages == 0:
        raise ScannedPDFError(
            "No meaningful embedded text was found. This appears to be a scanned "
            "PDF; run OCR on it before uploading."
        )
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end
        if target_end < len(text):
            boundary = max(text.rfind("\n", start + chunk_size // 2, target_end),
                           text.rfind(" ", start + chunk_size // 2, target_end))
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split pages separately so every chunk has an unambiguous page citation."""
    chunks: list[TextChunk] = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, overlap):
            index = len(chunks)
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}:{index}",
                    document_id=document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks


def ingest_pdf(
    filename: str,
    data: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestedDocument:
    """Validate, extract, and chunk a PDF."""
    pages = extract_pages(filename, data)
    document_id = hashlib.sha256(data).hexdigest()
    chunks = chunk_pages(pages, document_id, chunk_size, overlap)
    if not chunks:
        raise PDFIngestionError("The PDF did not produce any searchable text chunks.")
    return IngestedDocument(
        document_id=document_id,
        filename=pages[0].filename,
        page_count=len(pages),
        chunks=tuple(chunks),
    )


def ingest_multiple_pdfs(
    files: Sequence[tuple[str, bytes]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[IngestedDocument], list[tuple[str, str]]]:
    """Ingest multiple PDFs, returning (successful_documents, errors)."""
    documents: list[IngestedDocument] = []
    errors: list[tuple[str, str]] = []
    for filename, data in files:
        try:
            doc = ingest_pdf(filename, data, chunk_size, overlap)
            documents.append(doc)
        except Exception as exc:
            errors.append((filename, str(exc)))
    return documents, errors

