from document_rag_mcp.ingestion.extractor import DocumentExtractor

extractor = DocumentExtractor()
pages = extractor.extract("/home/jan/Notes/KiTa/Elternversammlung/2024-12-12 protokoll Elternvertreterversammlung.md")

for page in pages:
    print(f"Page {page.page_number} has {len(page.headings)} headings:")
    for h in page.headings:
        print(f"  Level {h[1]}: {h[0]}")
