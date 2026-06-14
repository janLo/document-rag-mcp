from datetime import datetime, timezone
from document_rag_mcp.storage.state_store import StateStore


def test_state_store_lifecycle(tmp_path):
    store = StateStore(tmp_path)
    
    file_path = str(tmp_path / "test.txt")
    collection = "my-collection"
    file_hash = "abc123hash"
    last_mod = datetime.now(timezone.utc).timestamp()
    
    # 1. Initially file should not exist
    assert store.get_file(file_path) is None
    
    # 2. Save file state with two chunks
    chunks = [
        ("chunk_0", "chunk_hash_0", 0),
        ("chunk_1", "chunk_hash_1", 1)
    ]
    store.save_file_state(file_path, collection, file_hash, last_mod, chunks)
    
    # 3. Retrieve and assert file metadata
    file_meta = store.get_file(file_path)
    assert file_meta is not None
    assert file_meta["file_path"] == file_path
    assert file_meta["collection"] == collection
    assert file_meta["file_hash"] == file_hash
    assert file_meta["chunk_count"] == 2
    assert file_meta["last_modified"] == last_mod
    
    # 4. Retrieve and assert chunks
    file_chunks = store.get_file_chunks(file_path)
    assert len(file_chunks) == 2
    assert file_chunks[0]["chunk_id"] == "chunk_0"
    assert file_chunks[0]["chunk_hash"] == "chunk_hash_0"
    assert file_chunks[0]["chunk_index"] == 0
    assert file_chunks[1]["chunk_id"] == "chunk_1"
    
    # 5. List all files
    all_files = store.list_all_files()
    assert len(all_files) == 1
    assert all_files[0]["file_path"] == file_path
    
    # 6. List collection files
    coll_files = store.list_collection_files(collection)
    assert len(coll_files) == 1
    assert coll_files[0]["file_path"] == file_path
    
    # List files of non-existent collection
    assert len(store.list_collection_files("empty")) == 0


def test_state_store_cascade_delete(tmp_path):
    store = StateStore(tmp_path)
    file_path = str(tmp_path / "test.txt")
    
    store.save_file_state(
        file_path,
        "collection",
        "hash",
        1.0,
        [("c0", "h0", 0)]
    )
    
    # Verify records in db
    assert store.get_file(file_path) is not None
    assert len(store.get_file_chunks(file_path)) == 1
    
    # Delete file
    store.delete_file(file_path)
    
    # Verify file is deleted
    assert store.get_file(file_path) is None
    # Verify chunks are also cascade deleted (via foreign key check)
    assert len(store.get_file_chunks(file_path)) == 0


def test_state_store_overwrite(tmp_path):
    store = StateStore(tmp_path)
    file_path = str(tmp_path / "test.txt")
    
    # Save first version
    store.save_file_state(
        file_path,
        "coll",
        "hash1",
        1.0,
        [("c0", "h0", 0)]
    )
    
    # Save updated version
    store.save_file_state(
        file_path,
        "coll",
        "hash2",
        2.0,
        [("c1", "h1", 0), ("c2", "h2", 1)]
    )
    
    file_meta = store.get_file(file_path)
    assert file_meta["file_hash"] == "hash2"
    assert file_meta["chunk_count"] == 2
    
    chunks = store.get_file_chunks(file_path)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "c1"
    assert chunks[1]["chunk_id"] == "c2"
