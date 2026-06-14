from datetime import datetime, timezone
import pytest
from document_rag_mcp.models import ChunkMetadata, DocumentChunk
from document_rag_mcp.storage.vector_store import VectorStore, sanitize_collection_name


def test_sanitize_collection_name():
    # Test valid names remain unchanged (or sanitized within rules)
    assert sanitize_collection_name("valid-name") == "valid-name"
    assert sanitize_collection_name("valid_name.1") == "valid_name.1"

    # Test length boundaries
    assert len(sanitize_collection_name("a")) == 3
    assert len(sanitize_collection_name("a" * 100)) == 63

    # Test invalid characters replaced by underscores
    assert sanitize_collection_name("my name!") == "my_namea"  # '!' at end gets padded/sanitized
    assert sanitize_collection_name("a@b#c$d") == "a_b_c_d"

    # Test start and end with alphanumeric
    assert sanitize_collection_name(".start") == "astart"
    assert sanitize_collection_name("end.") == "enda"

    # Test double dots
    assert sanitize_collection_name("abc..def") == "abc.def"


def test_vector_store_lifecycle(tmp_path):
    store = VectorStore(tmp_path)
    collection_name = "test-collection"

    # 1. ensure collection works
    store.ensure_collection(collection_name)
    assert sanitize_collection_name(collection_name) in store.list_collections()

    # 2. create document chunks with embeddings
    now = datetime.now(timezone.utc)
    meta = ChunkMetadata(
        file_path="/path/to/doc.txt",
        file_name="doc.txt",
        collection=collection_name,
        file_hash="file_hash_1",
        chunk_hash="chunk_hash_1",
        chunk_index=0,
        total_chunks=1,
        title="Test Document",
        section="Intro",
        file_type="txt",
        last_modified=now,
        ingested_at=now,
    )

    chunk = DocumentChunk(
        id="chunk_id_1",
        text="Hello world! This is a test chunk.",
        metadata=meta,
        embedding=[0.1] * 768,  # dummy embedding of size 768
    )

    # 3. upsert chunks
    store.upsert_chunks(collection_name, [chunk])
    assert store.collection_stats(collection_name)["count"] == 1

    # 4. Search
    results = store.search(collection_name, [0.1] * 768, top_k=5)
    assert len(results) == 1
    assert results[0].text == "Hello world! This is a test chunk."
    assert results[0].metadata.title == "Test Document"
    assert results[0].metadata.section == "Intro"
    # Ensure dates parsed back correctly
    assert isinstance(results[0].metadata.last_modified, datetime)

    # Search in non-existent collection should return empty
    assert len(store.search("missing-collection", [0.1] * 768)) == 0

    # 5. Delete by file path
    store.delete_by_file(collection_name, "/path/to/doc.txt")
    assert store.collection_stats(collection_name)["count"] == 0


def test_upsert_chunk_missing_embedding(tmp_path):
    store = VectorStore(tmp_path)
    meta = ChunkMetadata(
        file_path="/path/to/doc.txt",
        file_name="doc.txt",
        collection="coll",
        file_hash="hash",
        chunk_hash="chash",
        chunk_index=0,
        total_chunks=1,
        file_type="txt",
        last_modified=datetime.now(timezone.utc),
        ingested_at=datetime.now(timezone.utc),
    )
    chunk = DocumentChunk(
        id="chunk_id_1",
        text="hello",
        metadata=meta,
        embedding=None,  # missing embedding
    )
    with pytest.raises(ValueError, match="does not have an embedding"):
        store.upsert_chunks("coll", [chunk])
