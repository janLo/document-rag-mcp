from unittest.mock import AsyncMock, MagicMock
import pytest
from document_rag_mcp.config import EmbeddingConfig
from document_rag_mcp.embedding.client import EmbeddingClient


@pytest.mark.asyncio
async def test_embedding_client_embed():
    config = EmbeddingConfig(batch_size=2, dimensions=768)
    client = EmbeddingClient(config)

    mock_create = AsyncMock()
    client.client.embeddings.create = mock_create

    # Create dummy response structure
    mock_data1 = MagicMock()
    mock_data1.embedding = [0.1] * 768
    mock_data2 = MagicMock()
    mock_data2.embedding = [0.2] * 768

    mock_response1 = MagicMock()
    mock_response1.data = [mock_data1, mock_data2]
    
    mock_data3 = MagicMock()
    mock_data3.embedding = [0.3] * 768
    mock_response2 = MagicMock()
    mock_response2.data = [mock_data3]

    mock_create.side_effect = [mock_response1, mock_response2]

    texts = ["hello", "world", "test"]
    results = await client.embed(texts)

    assert len(results) == 3
    assert results[0] == [0.1] * 768
    assert results[1] == [0.2] * 768
    assert results[2] == [0.3] * 768

    assert mock_create.call_count == 2
    # Verify dimensions was passed
    mock_create.assert_any_call(input=["hello", "world"], model=config.model, dimensions=768)
    mock_create.assert_any_call(input=["test"], model=config.model, dimensions=768)


@pytest.mark.asyncio
async def test_embedding_client_fallback():
    config = EmbeddingConfig(batch_size=1, dimensions=768)
    client = EmbeddingClient(config)

    mock_create = AsyncMock()
    client.client.embeddings.create = mock_create

    mock_data = MagicMock()
    mock_data.embedding = [0.5] * 768
    mock_response = MagicMock()
    mock_response.data = [mock_data]

    # First call fails (raising exception on dimensions), second succeeds
    mock_create.side_effect = [Exception("API does not support dimensions"), mock_response]

    results = await client.embed(["hello"])
    assert len(results) == 1
    assert results[0] == [0.5] * 768
    assert mock_create.call_count == 2

    # Check call arguments
    args1, kwargs1 = mock_create.call_args_list[0]
    assert kwargs1["dimensions"] == 768

    args2, kwargs2 = mock_create.call_args_list[1]
    assert "dimensions" not in kwargs2


@pytest.mark.asyncio
async def test_embed_query():
    config = EmbeddingConfig(batch_size=1, dimensions=768)
    client = EmbeddingClient(config)

    mock_create = AsyncMock()
    client.client.embeddings.create = mock_create

    mock_data = MagicMock()
    mock_data.embedding = [0.9] * 768
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_create.return_value = mock_response

    result = await client.embed_query("search me")
    assert result == [0.9] * 768
    mock_create.assert_called_once()
