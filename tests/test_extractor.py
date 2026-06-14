from pathlib import Path
import fitz
import pytest
from document_rag_mcp.ingestion.extractor import DocumentExtractor


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "sample.txt"
    content = "Welcome to the Project\n\nThis is a simple text file with some text."
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def md_file(tmp_path):
    path = tmp_path / "sample.md"
    content = """---
title: Custom Title
author: Jane Doe
---
# Main Heading

Some introductory text.

## Section 1

More detailed content here.
"""
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()

    # Page 1: Normal text with headings
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72), "Document Title", fontsize=18, fontname="hebo"
    )
    page1.insert_text(
        (72, 100), "This is a subheading", fontsize=14, fontname="hebo"
    )
    page1.insert_text(
        (72, 130),
        "This is the body text. It contains a lot of regular words.",
        fontsize=10,
        fontname="helv",
    )
    page1.insert_text(
        (72, 150),
        "More body text here to establish the dominant font size.",
        fontsize=10,
        fontname="helv",
    )
    page1.insert_text(
        (72, 170),
        "Yet another sentence of body text in the PDF file.",
        fontsize=10,
        fontname="helv",
    )

    # Page 2: Blank / Scanned page (draw a simple line instead of text to simulate image/no text)
    page2 = doc.new_page()
    page2.draw_line((50, 50), (200, 200), color=(1, 0, 0), width=3)

    doc.save(path)
    doc.close()
    return path


def test_extract_txt(txt_file):
    extractor = DocumentExtractor()
    pages = extractor.extract(txt_file)

    assert len(pages) == 1
    assert "Welcome to the Project" in pages[0].text
    assert len(pages[0].headings) == 1
    assert pages[0].headings[0] == ("Welcome to the Project", 1)
    assert pages[0].page_number == 1
    assert pages[0].metadata["title"] == "sample"


def test_extract_md(md_file):
    extractor = DocumentExtractor()
    pages = extractor.extract(md_file)

    assert len(pages) == 1
    assert "Main Heading" in pages[0].text
    assert "# Main Heading" in pages[0].text  # Should preserve headers for structure-aware chunkers
    assert len(pages[0].headings) == 2
    assert pages[0].headings[0] == ("Main Heading", 1)
    assert pages[0].headings[1] == ("Section 1", 2)
    assert pages[0].metadata["title"] == "Custom Title"
    assert pages[0].metadata["frontmatter"]["author"] == "Jane Doe"


def test_extract_pdf_no_vision(pdf_file):
    # Vision disabled -> should extract text, scanned page gets no image bytes
    extractor = DocumentExtractor(vision_enabled=False)
    pages = extractor.extract(pdf_file)

    assert len(pages) == 2
    assert "Document Title" in pages[0].text
    assert "body text" in pages[0].text
    assert len(pages[0].headings) >= 2
    assert pages[0].headings[0] == ("Document Title", 1)

    assert pages[1].text.strip() == ""
    assert pages[1].image_bytes is None


def test_extract_pdf_with_vision(pdf_file):
    # Vision enabled -> should render empty/scanned pages to images
    extractor = DocumentExtractor(vision_enabled=True)
    pages = extractor.extract(pdf_file)

    assert len(pages) == 2
    assert pages[0].image_bytes is None  # Page 1 has text, not scanned

    assert pages[1].text.strip() == ""
    assert pages[1].image_bytes is not None  # Page 2 scanned -> rendered
    assert isinstance(pages[1].image_bytes, bytes)
    assert pages[1].image_bytes.startswith(b"\x89PNG\r\n\x1a\n")  # valid PNG header


def test_extract_invalid_extension():
    extractor = DocumentExtractor()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        extractor.extract(Path("test.invalid"))
