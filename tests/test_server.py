from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from document_rag_mcp.config import CollectionConfig
from document_rag_mcp.models import ChunkMetadata, SearchResult


# Import server after conftest has configured the temporary environment
import document_rag_mcp.server as server


def test_is_safe_path():
    collections = [
        CollectionConfig(name="docs", paths=[Path("/workspace/my-docs")]),
        CollectionConfig(name="files", paths=[Path("/workspace/single-file.txt")]),
    ]

    # Nested file
    assert server.is_safe_path("/workspace/my-docs/sub/folder/file.txt", collections) is True
    # Exact direct file match
    assert server.is_safe_path("/workspace/single-file.txt", collections) is True

    # Traversals / Outside paths
    assert server.is_safe_path("/workspace/my-docs/../../etc/passwd", collections) is False
    assert server.is_safe_path("/workspace/other-folder/file.txt", collections) is False
    assert server.is_safe_path("/workspace/single-file.txt.bak", collections) is False


@pytest.mark.asyncio
async def test_server_tool_search():
    # Setup mocks
    mock_search = AsyncMock()
    server.search_engine.search = mock_search

    now = datetime.now(timezone.utc)
    meta = ChunkMetadata(
        file_path="/workspace/my-docs/file.txt",
        file_name="file.txt",
        collection="docs",
        file_hash="hash",
        chunk_hash="chash",
        chunk_index=0,
        total_chunks=1,
        title="Document Title",
        section="Intro",
        file_type="txt",
        last_modified=now,
        ingested_at=now,
    )
    mock_search.return_value = [SearchResult(text="matching chunk content", score=0.1, metadata=meta)]

    res = await server.search(query="test query", collection="docs", top_k=2)

    assert "matching chunk content" in res
    assert "Document Title" in res
    assert "0.1000" in res  # Score formatted
    mock_search.assert_called_once_with("test query", "docs", 2)


@pytest.mark.asyncio
async def test_server_tool_get_document_info():
    mock_get_file = MagicMock()
    server.state_store.get_file = mock_get_file

    # File not tracked
    mock_get_file.return_value = None
    res = await server.get_document_info("/docs/doc.txt")
    assert "not indexed" in res

    # File tracked
    mock_get_file.return_value = {
        "file_path": "/docs/doc.txt",
        "collection": "docs",
        "file_hash": "hash123",
        "chunk_count": 5,
        "last_modified": 1234567.0,
        "ingested_at": "2026-06-14T12:00:00Z",
    }
    res2 = await server.get_document_info("/docs/doc.txt")
    assert "hash123" in res2
    assert "5" in res2


@pytest.mark.asyncio
async def test_server_tool_get_document_content(tmp_path):
    # Setup mock collections to authorize tmp_path folder
    server.config.collections = [CollectionConfig(name="test", paths=[tmp_path])]

    test_file = tmp_path / "safe.txt"
    test_file.write_text("Secret text content inside the document.", encoding="utf-8")

    # Safe access
    res = await server.get_document_content(str(test_file))
    assert "Secret text content" in res

    # Unsafe access (traversal outside collection folders)
    res_denied = await server.get_document_content("/etc/passwd")
    assert "Access Denied" in res_denied


@pytest.mark.asyncio
async def test_server_tool_get_document_original(tmp_path):
    server.config.collections = [CollectionConfig(name="test", paths=[tmp_path])]

    test_file = tmp_path / "binary.pdf"
    test_file.write_bytes(b"%PDF-1.4 binary data")

    # Safe access
    res = await server.get_document_original(str(test_file))
    assert res == b"%PDF-1.4 binary data"

    # Unsafe access raises PermissionError
    with pytest.raises(PermissionError, match="Access Denied"):
        await server.get_document_original("/etc/passwd")


@pytest.mark.asyncio
async def test_server_tool_ingest_now(tmp_path):
    # Setup collection
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "a.txt").write_text("content")

    server.config.collections = [CollectionConfig(name="docs", paths=[doc_dir])]

    # Mock pipeline and scanner
    mock_ingest = AsyncMock(return_value=True)
    server.pipeline.ingest_file = mock_ingest

    # Also mock state store file listing to simulate no deleted files
    server.state_store.list_collection_files = MagicMock(return_value=[])

    res = await server.ingest_now(collection="docs")
    assert "Ingested/Updated: 1" in res
    mock_ingest.assert_called_once()


@pytest.mark.asyncio
async def test_server_resources():
    # Setup fake collection stats and info
    server.config.collections = [
        CollectionConfig(name="c1", paths=[Path("/p1")], file_patterns=["*.txt"])
    ]
    server.vector_store.collection_stats = MagicMock(return_value={"count": 10})
    server.state_store.list_collection_files = MagicMock(
        return_value=[{"file_path": "/p1/a.txt", "file_hash": "h1", "last_modified": 1.0}]
    )

    # 1. list resource
    res_list = await server.collections_resource()
    parsed_list = json.loads(res_list)
    assert len(parsed_list) == 1
    assert parsed_list[0]["name"] == "c1"
    assert parsed_list[0]["indexed_chunks"] == 10

    # 2. info resource
    res_info = await server.collection_info_resource("c1")
    parsed_info = json.loads(res_info)
    assert parsed_info["name"] == "c1"
    assert len(parsed_info["files"]) == 1
    assert parsed_info["files"][0]["file_path"] == "/p1/a.txt"

    # 3. stats resource
    res_stats = await server.collection_stats_resource("c1")
    parsed_stats = json.loads(res_stats)
    assert parsed_stats["count"] == 10
