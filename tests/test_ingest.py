import pymupdf as fitz
import pytest

from ingest import (
    PDFIngestionError,
    PageText,
    ScannedPDFError,
    chunk_pages,
    extract_pages,
    ingest_multiple_pdfs,
    ingest_pdf,
)


def make_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_ingest_retains_filename_and_page_numbers():
    data = make_pdf([
        "Alpha policy details " * 20,
        "Beta procedure details " * 20,
    ])

    result = ingest_pdf("../handbook.pdf", data, chunk_size=160, overlap=30)

    assert result.filename == "handbook.pdf"
    assert result.page_count == 2
    assert {chunk.page_number for chunk in result.chunks} == {1, 2}
    assert all(chunk.document_id == result.document_id for chunk in result.chunks)


def test_chunks_overlap_and_have_unique_ids():
    pages = [PageText("notes.pdf", 1, "word " * 150)]
    chunks = chunk_pages(pages, "doc123", chunk_size=120, overlap=20)

    assert len(chunks) > 2
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.filename == "notes.pdf" for chunk in chunks)


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("notes.txt", b"%PDF-1.7", ".pdf extension"),
        ("notes.pdf", b"not a pdf", "valid PDF signature"),
        ("notes.pdf", b"", "empty"),
    ],
)
def test_invalid_uploads_are_rejected(filename, data, message):
    with pytest.raises(PDFIngestionError, match=message):
        ingest_pdf(filename, data)


def test_image_only_pdf_is_reported_as_scanned():
    with pytest.raises(ScannedPDFError, match="scanned"):
        ingest_pdf("scan.pdf", make_pdf([""]))


def test_chunk_parameters_are_validated():
    pages = extract_pages("notes.pdf", make_pdf(["useful text " * 20]))
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(pages, "doc", chunk_size=100, overlap=100)


def test_ingest_multiple_pdfs_with_partial_failures():
    valid_pdf_1 = make_pdf(["Document one policy details " * 20])
    valid_pdf_2 = make_pdf(["Document two guidelines details " * 20])
    invalid_pdf = b"not a valid pdf"

    files = [
        ("doc1.pdf", valid_pdf_1),
        ("broken.pdf", invalid_pdf),
        ("doc2.pdf", valid_pdf_2),
    ]

    docs, errors = ingest_multiple_pdfs(files)

    assert len(docs) == 2
    assert {d.filename for d in docs} == {"doc1.pdf", "doc2.pdf"}
    assert len(errors) == 1
    assert errors[0][0] == "broken.pdf"
    assert "valid PDF signature" in errors[0][1]

