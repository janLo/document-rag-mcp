from unittest.mock import patch
import pytest
from watchfiles import Change
from document_rag_mcp.config import CollectionConfig
from document_rag_mcp.ingestion.watcher import resolve_collection, watch_collections


def test_resolve_collection(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    coll = CollectionConfig(
        name="test-coll",
        paths=[docs_dir],
        file_patterns=["*.txt"],
    )

    # 1. Matching file
    file_path = docs_dir / "file.txt"
    res = resolve_collection(file_path, [coll])
    assert res is not None
    assert res[0].name == "test-coll"
    assert res[1] == docs_dir.absolute()

    # 2. File mismatch extension
    file_path2 = docs_dir / "file.md"
    assert resolve_collection(file_path2, [coll]) is None

    # 3. File in hidden folder
    hidden_file = docs_dir / ".git" / "file.txt"
    assert resolve_collection(hidden_file, [coll]) is None

    # 4. Direct file configuration
    direct_file = tmp_path / "direct.txt"
    direct_file.write_text("direct")
    
    coll_direct = CollectionConfig(
        name="direct-coll",
        paths=[direct_file],
        file_patterns=["*.txt"],
    )
    
    res = resolve_collection(direct_file, [coll_direct])
    assert res is not None
    assert res[0].name == "direct-coll"


@pytest.mark.asyncio
async def test_watch_collections(tmp_path):
    coll = CollectionConfig(
        name="test-coll",
        paths=[tmp_path],
        file_patterns=["*.txt"],
    )

    # Mock awatch as an async generator yielding one mock change set
    async def mock_awatch_generator(*paths, **kwargs):
        yield {(Change.added, str(tmp_path / "new.txt"))}
        # wait forever to keep loop alive or let it raise GeneratorExit
        await asyncio.sleep(1)

    import asyncio

    with patch("document_rag_mcp.ingestion.watcher.awatch", side_effect=mock_awatch_generator):
        callback_calls = []

        async def callback(change_type, path, collection_name):
            callback_calls.append((change_type, path, collection_name))
            # Raise exception to break out of the infinite async loop for test
            raise GeneratorExit("Stop loop")

        try:
            await watch_collections([coll], callback)
        except GeneratorExit:
            pass

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("added", (tmp_path / "new.txt").resolve(), "test-coll")
