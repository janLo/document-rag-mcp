from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest
from document_rag_mcp.models import ChunkMetadata, DocumentChunk
from document_rag_mcp.search.engine import SearchEngine
from document_rag_mcp.storage.vector_store import VectorStore


@pytest.fixture
def mock_embedding_client():
    client = MagicMock()
    async def embed_query_mock(query):
        if "one" in query:
            return [0.1] * 768
        return [0.5] * 768
    client.embed_query = embed_query_mock
    return client


@pytest.mark.asyncio
async def test_search_single_collection(tmp_path, mock_embedding_client):
    store = VectorStore(tmp_path)
    engine = SearchEngine(store, mock_embedding_client)
    
    # Setup collection
    store.ensure_collection("coll1")
    now = datetime.now(timezone.utc)
    
    meta = ChunkMetadata(
        file_path="/docs/a.txt",
        file_name="a.txt",
        collection="coll1",
        file_hash="hash",
        chunk_hash="chash",
        chunk_index=0,
        total_chunks=1,
        file_type="txt",
        last_modified=now,
        ingested_at=now
    )
    
    chunk = DocumentChunk(
        id="c1",
        text="Unique chunk text one",
        metadata=meta,
        embedding=[0.1] * 768
    )
    store.upsert_chunks("coll1", [chunk])
    
    results = await engine.search("find one", collection_name="coll1", top_k=2)
    assert len(results) == 1
    assert results[0].text == "Unique chunk text one"
    assert results[0].metadata.collection == "coll1"


@pytest.mark.asyncio
async def test_search_multi_collection(tmp_path, mock_embedding_client):
    store = VectorStore(tmp_path)
    engine = SearchEngine(store, mock_embedding_client)
    
    # Setup two collections
    store.ensure_collection("coll1")
    store.ensure_collection("coll2")
    
    now = datetime.now(timezone.utc)
    
    # Chunk in collection 1 (higher distance)
    meta1 = ChunkMetadata(
        file_path="/docs/a.txt", file_name="a.txt", collection="coll1",
        file_hash="h1", chunk_hash="c_h1", chunk_index=0, total_chunks=1,
        file_type="txt", last_modified=now, ingested_at=now
    )
    chunk1 = DocumentChunk(
        id="c1", text="text from coll1", metadata=meta1,
        embedding=[0.2] * 768
    )
    
    # Chunk in collection 2 (lower distance - closer to query [0.1]*768)
    meta2 = ChunkMetadata(
        file_path="/docs/b.txt", file_name="b.txt", collection="coll2",
        file_hash="h2", chunk_hash="c_h2", chunk_index=0, total_chunks=1,
        file_type="txt", last_modified=now, ingested_at=now
    )
    chunk2 = DocumentChunk(
        id="c2", text="text from coll2", metadata=meta2,
        embedding=[0.11] * 768
    )
    
    store.upsert_chunks("coll1", [chunk1])
    store.upsert_chunks("coll2", [chunk2])
    
    # Query: [0.1] * 768
    # chunk2 is closer (dist ~ (0.11-0.1)^2 = 0.0001 per dim)
    # chunk1 is further (dist ~ (0.2-0.1)^2 = 0.01 per dim)
    results = await engine.search("find one", collection_name=None, top_k=5)
    
    assert len(results) == 2
    # The result with smaller distance (coll2) should be ranked first!
    assert results[0].text == "text from coll2"
    assert results[0].metadata.collection == "coll2"
    assert results[1].text == "text from coll1"
    assert results[1].metadata.collection == "coll1"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_search_no_collections(tmp_path, mock_embedding_client):
    # Empty vector store with no collections
    store = VectorStore(tmp_path)
    engine = SearchEngine(store, mock_embedding_client)
    
    results = await engine.search("any query", collection_name=None)
    assert len(results) == 0
