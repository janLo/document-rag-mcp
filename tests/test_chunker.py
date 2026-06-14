from datetime import datetime, timezone
from document_rag_mcp.config import ChunkingConfig
from document_rag_mcp.ingestion.chunker import DocumentChunker
from document_rag_mcp.ingestion.extractor import ExtractedPage


def test_chunker_fallback():
    # Attempt to initialize with a dummy/invalid local model to trigger fallback
    config = ChunkingConfig(local_model="invalid-local-model")
    chunker = DocumentChunker(config)

    # Chunker initialization should not fail; it should fallback to TokenChunker
    assert chunker.semantic_chunker is not None

    pages = [ExtractedPage(text="This is simple text.", page_number=1)]
    chunks = chunker.chunk_pages(
        pages,
        file_path="/docs/test.txt",
        collection_name="coll",
        file_hash="hash",
        file_type="txt",
        last_modified=datetime.now(timezone.utc),
    )
    assert len(chunks) == 1
    assert chunks[0].text == "This is simple text."
    assert chunks[0].metadata.file_name == "test.txt"


def test_chunker_markdown():
    config = ChunkingConfig(max_chunk_size=100)
    chunker = DocumentChunker(config)

    # Page with frontmatter title resolved, headings and text
    pages = [
        ExtractedPage(
            text="# Introduction\nThis is paragraph one.\n\n## Details\nThis is paragraph two.",
            page_number=1,
            headings=[("Introduction", 1), ("Details", 2)],
            metadata={"title": "Custom Doc Title"},
        )
    ]

    chunks = chunker.chunk_pages(
        pages,
        file_path="/docs/doc.md",
        collection_name="md-coll",
        file_hash="hash",
        file_type="md",
        last_modified=datetime.now(timezone.utc),
    )

    assert len(chunks) >= 1
    # Check inheritance of title
    assert chunks[0].metadata.title == "Custom Doc Title"
    # Check section resolution (should match the nearest heading)
    assert chunks[0].metadata.section in ["Introduction", "Details"]
    assert chunks[0].metadata.file_type == "md"


def test_chunker_pdf_page_numbers():
    config = ChunkingConfig(max_chunk_size=100)
    chunker = DocumentChunker(config)

    pages = [
        ExtractedPage(text="Page one content.", page_number=1, headings=[("Page One Heading", 1)]),
        ExtractedPage(text="Page two content.", page_number=2, headings=[("Page Two Heading", 1)]),
    ]

    chunks = chunker.chunk_pages(
        pages,
        file_path="/docs/doc.pdf",
        collection_name="pdf-coll",
        file_hash="hash",
        file_type="pdf",
        last_modified=datetime.now(timezone.utc),
    )

    # We should have chunks for both pages
    assert len(chunks) >= 2
    
    # Verify page numbers mapped correctly
    page_numbers = [c.metadata.page_number for c in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers

    # Verify document titles are correct
    for c in chunks:
        assert c.metadata.title == "Page One Heading"
