from document_rag_mcp.config import CollectionConfig
from document_rag_mcp.ingestion.scanner import scan_all_collections, scan_collection_files


def test_scanner_basic(tmp_path):
    # Setup files
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    
    file1 = doc_dir / "doc1.txt"
    file1.write_text("content1")
    
    file2 = doc_dir / "doc2.md"
    file2.write_text("content2")
    
    # Nested folder
    sub_dir = doc_dir / "sub"
    sub_dir.mkdir()
    file3 = sub_dir / "doc3.pdf"
    file3.write_text("content3")
    
    # Excluded patterns or names
    file4 = doc_dir / "image.png"
    file4.write_text("png")
    
    # Hidden folder to be ignored
    hidden_dir = doc_dir / ".hidden"
    hidden_dir.mkdir()
    file5 = hidden_dir / "hidden.txt"
    file5.write_text("hidden")

    # Collection configuration
    config = CollectionConfig(
        name="test-coll",
        paths=[doc_dir],
        file_patterns=["*.txt", "*.md", "*.pdf"]
    )
    
    # Scan files
    files = scan_collection_files(config)
    
    # Assert correct matching and resolving
    assert len(files) == 3
    assert file1.resolve() in files
    assert file2.resolve() in files
    assert file3.resolve() in files
    assert file4.resolve() not in files
    assert file5.resolve() not in files  # hidden files ignored


def test_scanner_direct_file(tmp_path):
    file_path = tmp_path / "single.txt"
    file_path.write_text("single text")
    
    config = CollectionConfig(
        name="test-coll",
        paths=[file_path],
        file_patterns=["*.txt"]
    )
    
    files = scan_collection_files(config)
    assert len(files) == 1
    assert files[0] == file_path.resolve()


def test_scan_all_collections(tmp_path):
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    f1 = dir1 / "a.md"
    f1.write_text("a")
    
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    f2 = dir2 / "b.txt"
    f2.write_text("b")
    
    c1 = CollectionConfig(name="coll1", paths=[dir1], file_patterns=["*.md"])
    c2 = CollectionConfig(name="coll2", paths=[dir2], file_patterns=["*.txt"])
    
    results = scan_all_collections([c1, c2])
    
    assert "coll1" in results
    assert "coll2" in results
    assert len(results["coll1"]) == 1
    assert results["coll1"][0] == f1.resolve()
    assert len(results["coll2"]) == 1
    assert results["coll2"][0] == f2.resolve()
