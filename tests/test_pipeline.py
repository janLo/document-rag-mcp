from pathlib import Path
from unittest.mock import MagicMock
import fitz
import pytest
from document_rag_mcp.config import AppConfig
from document_rag_mcp.ingestion.pipeline import IngestionPipeline
from document_rag_mcp.storage.state_store import StateStore
from document_rag_mcp.storage.vector_store import VectorStore


@pytest.fixture
def mock_embedding_client():
    client = MagicMock()

    async def embed_mock(texts):
        return [[0.1] * 768 for _ in texts]

    client.embed = embed_mock
    return client


@pytest.fixture
def mock_vision_client():
    client = MagicMock()

    async def extract_mock(image_bytes):
        return "OCR Text: From image scan"

    client.extract_text_from_image = extract_mock
    return client


@pytest.mark.asyncio
async def test_pipeline_new_and_unchanged_file(
    tmp_path, mock_embedding_client, mock_vision_client
):
    config = AppConfig()
    state_store = StateStore(tmp_path)
    vector_store = VectorStore(tmp_path)

    pipeline = IngestionPipeline(
        config=config,
        state_store=state_store,
        vector_store=vector_store,
        embedding_client=mock_embedding_client,
        vision_client=mock_vision_client,
    )

    test_file = tmp_path / "doc.txt"
    test_file.write_text("Hello. This is a text file.", encoding="utf-8")

    # 1. New file should be ingested
    ingested = await pipeline.ingest_file(test_file, "coll1")
    assert ingested is True

    file_meta = state_store.get_file(str(test_file))
    assert file_meta is not None
    assert file_meta["collection"] == "coll1"
    assert file_meta["chunk_count"] == 1

    chunks = state_store.get_file_chunks(str(test_file))
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0

    # Verify vector store has it
    results = vector_store.search("coll1", [0.1] * 768)
    assert len(results) == 1
    assert results[0].text == "Hello. This is a text file."

    # 2. Ingest again (unchanged) -> should skip (return False)
    ingested_again = await pipeline.ingest_file(test_file, "coll1")
    assert ingested_again is False


def create_test_pdf(path: Path, page1_text: str, page2_text: str):
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), page1_text, fontsize=10, fontname="helv")
    p2 = doc.new_page()
    p2.insert_text((72, 72), page2_text, fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()


@pytest.mark.asyncio
async def test_pipeline_incremental_chunk_dedup(tmp_path, mock_vision_client):
    config = AppConfig()
    config.chunking.local_model = "invalid-local-model"
    config.chunking.max_chunk_size = 12
    config.chunking.chunk_overlap = 0
    state_store = StateStore(tmp_path)
    vector_store = VectorStore(tmp_path)

    # Count how many texts are embedded
    embed_calls = 0

    async def mock_embed(texts):
        nonlocal embed_calls
        embed_calls += len(texts)
        return [[0.2] * 768 for _ in texts]

    mock_emb_client = MagicMock()
    mock_emb_client.embed = mock_embed

    pipeline = IngestionPipeline(
        config=config,
        state_store=state_store,
        vector_store=vector_store,
        embedding_client=mock_emb_client,
        vision_client=mock_vision_client,
    )

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("First long text block here. Second long text block here.", encoding="utf-8")

    # Ingest 1: Index the file
    await pipeline.ingest_file(txt_path, "coll")
    first_call_count = embed_calls
    assert first_call_count > 0, "Should have embedded some chunks"

    # Verify collection counts
    assert vector_store.collection_stats("coll")["count"] == first_call_count

    # Ingest 2: Modify first chunk, keep second chunk identical
    txt_path.write_text("Third long text block here. Second long text block here.", encoding="utf-8")
    await pipeline.ingest_file(txt_path, "coll")

    # Only the modified chunk(s) should be embedded. Unmodified chunks should reuse their vectors.
    # Total embed calls should be less than a full double run.
    assert first_call_count < embed_calls < first_call_count * 2
    assert vector_store.collection_stats("coll")["count"] == first_call_count


@pytest.mark.asyncio
async def test_pipeline_vision_ocr(tmp_path, mock_embedding_client, mock_vision_client):
    # Vision enabled
    config = AppConfig()
    config.vision.enabled = True

    state_store = StateStore(tmp_path)
    vector_store = VectorStore(tmp_path)

    pipeline = IngestionPipeline(
        config=config,
        state_store=state_store,
        vector_store=vector_store,
        embedding_client=mock_embedding_client,
        vision_client=mock_vision_client,
    )

    # Create a scanned PDF page (empty text, just drawing)
    pdf_path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.draw_line((10, 10), (100, 100))
    doc.save(pdf_path)
    doc.close()

    # Ingest scanned PDF
    await pipeline.ingest_file(pdf_path, "coll")

    # Chunks should contain the text extracted via mock vision client
    results = vector_store.search("coll", [0.1] * 768)
    assert len(results) == 1
    assert "OCR Text: From image scan" in results[0].text


@pytest.mark.asyncio
async def test_pipeline_delete_file(tmp_path, mock_embedding_client, mock_vision_client):
    config = AppConfig()
    state_store = StateStore(tmp_path)
    vector_store = VectorStore(tmp_path)

    pipeline = IngestionPipeline(
        config=config,
        state_store=state_store,
        vector_store=vector_store,
        embedding_client=mock_embedding_client,
        vision_client=mock_vision_client,
    )

    test_file = tmp_path / "delete_me.txt"
    test_file.write_text("This text will be deleted.", encoding="utf-8")

    await pipeline.ingest_file(test_file, "coll")
    assert vector_store.collection_stats("coll")["count"] == 1
    assert state_store.get_file(str(test_file)) is not None

    # Delete
    await pipeline.delete_file(str(test_file), "coll")

    # Verify pruned from both databases
    assert vector_store.collection_stats("coll")["count"] == 0
    assert state_store.get_file(str(test_file)) is None
