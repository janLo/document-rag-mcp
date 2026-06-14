from pathlib import Path
from unittest.mock import MagicMock
import fitz
import pytest
from document_rag_mcp.config import AppConfig, CollectionConfig
from document_rag_mcp.embedding.client import EmbeddingClient
from document_rag_mcp.ingestion.pipeline import IngestionPipeline
from document_rag_mcp.search.engine import SearchEngine
from document_rag_mcp.storage.state_store import StateStore
from document_rag_mcp.storage.vector_store import VectorStore
from document_rag_mcp.vision.client import VisionClient


def create_pdf(path: Path, p1: str, p2: str):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), p1, fontsize=10, fontname="helv")
    doc.new_page().insert_text((72, 72), p2, fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()


@pytest.mark.asyncio
async def test_integration_flow(tmp_path):
    # 1. Setup folders
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # 2. Setup config
    config = AppConfig()
    config.storage.data_dir = tmp_path / "data"
    config.collections = [
        CollectionConfig(
            name="test-coll",
            paths=[docs_dir],
            file_patterns=["*.txt", "*.md", "*.pdf"]
        )
    ]
    
    # 3. Setup mock embedding client
    embed_calls = 0
    async def mock_embed(texts):
        nonlocal embed_calls
        embed_calls += len(texts)
        # Mock simple embedding: [val] * 768 where val corresponds to text content
        vectors = []
        for t in texts:
            val = 0.1
            if "important" in t.lower():
                val = 0.9
            elif "target" in t.lower():
                val = 0.8
            vectors.append([val] * 768)
        return vectors
        
    async def mock_embed_query(query):
        val = 0.1
        if "important" in query.lower():
            val = 0.9
        elif "target" in query.lower():
            val = 0.8
        return [val] * 768

    mock_emb_client = MagicMock(spec=EmbeddingClient)
    mock_emb_client.embed = mock_embed
    mock_emb_client.embed_query = mock_embed_query
    
    # Mock vision
    mock_vis_client = MagicMock(spec=VisionClient)
    
    # 4. Instantiate modules
    state_store = StateStore(config.storage.data_dir)
    vector_store = VectorStore(config.storage.data_dir)
    pipeline = IngestionPipeline(
        config=config,
        state_store=state_store,
        vector_store=vector_store,
        embedding_client=mock_emb_client,
        vision_client=mock_vis_client
    )
    search_engine = SearchEngine(vector_store, mock_emb_client)

    # --- Step 1: Create initial documents ---
    txt_file = docs_dir / "sample.txt"
    txt_file.write_text("Hello integration test. This is an important target.", encoding="utf-8")
    
    md_file = docs_dir / "sample.md"
    md_file.write_text("# Doc\n\nSome plain markdown paragraph text.", encoding="utf-8")
    
    pdf_file = docs_dir / "sample.pdf"
    create_pdf(pdf_file, "Page one contains important target data.", "Page two has secondary information.")

    # --- Step 2: Synchronous Indexing ---
    from document_rag_mcp.ingestion.scanner import scan_collection_files
    files = scan_collection_files(config.collections[0])
    
    assert len(files) == 3
    
    # Run ingestion pipeline
    for f in files:
        await pipeline.ingest_file(f, "test-coll")
        
    # Verify in DB
    # PDF should have 2 chunks, MD 1 chunk, TXT 1 chunk (since they fit in default sizes)
    # Total chunks = 4
    assert vector_store.collection_stats("test-coll")["count"] == 4
    assert state_store.get_file(str(txt_file)) is not None
    assert state_store.get_file(str(md_file)) is not None
    assert state_store.get_file(str(pdf_file)) is not None
    
    first_ingest_calls = embed_calls
    assert first_ingest_calls == 4

    # --- Step 3: Run Semantic Search ---
    # Search for "important"
    results = await search_engine.search("important target", collection_name="test-coll", top_k=2)
    assert len(results) == 2
    # The chunks containing "important" or "target" should be closest and have score around 0 (L2 distance)
    assert "important" in results[0].text.lower() or "target" in results[0].text.lower()
    
    # --- Step 4: Incremental Update (modify one file) ---
    # We modify only the PDF page 1, leaving page 2 completely identical
    create_pdf(pdf_file, "Page one modified target information.", "Page two has secondary information.")
    
    # Run pipeline on PDF file again
    await pipeline.ingest_file(pdf_file, "test-coll")
    
    # Chunks count in Chroma should still be 4
    assert vector_store.collection_stats("test-coll")["count"] == 4
    
    # Check embedding calls: only 1 new page chunk should have been embedded,
    # the second page chunk was identical so its embedding was reused!
    # Total calls should be 5 (4 from run 1, 1 from run 2)
    assert embed_calls == 5

    # --- Step 5: Deletion and Pruning ---
    # Delete the txt file from disk
    txt_file.unlink()
    
    # Run pipeline on deleted file (or simulate ingest scan)
    # When ingest_file runs on non-existent path, it deletes it
    await pipeline.ingest_file(txt_file, "test-coll")
    
    # verify txt pruned
    assert state_store.get_file(str(txt_file)) is None
    # total chunks in Chroma should decrease from 4 to 3
    assert vector_store.collection_stats("test-coll")["count"] == 3
