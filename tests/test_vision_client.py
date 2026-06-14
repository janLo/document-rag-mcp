import base64
from unittest.mock import AsyncMock, MagicMock
import pytest
from document_rag_mcp.config import VisionConfig
from document_rag_mcp.vision.client import VisionClient


@pytest.mark.asyncio
async def test_vision_client_disabled():
    config = VisionConfig(enabled=False)
    client = VisionClient(config)

    # Should return empty string without calling OpenAI API
    text = await client.extract_text_from_image(b"fake_image_bytes")
    assert text == ""


@pytest.mark.asyncio
async def test_vision_client_extract():
    config = VisionConfig(enabled=True, model="gpt-4o")
    client = VisionClient(config)

    mock_create = AsyncMock()
    client.client.chat.completions.create = mock_create

    # Mock response structure
    mock_choice = MagicMock()
    mock_choice.message.content = "Extracted text content from chart"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_create.return_value = mock_response

    image_bytes = b"fake_png_bytes"
    expected_b64 = base64.b64encode(image_bytes).decode("utf-8")

    result = await client.extract_text_from_image(image_bytes)
    assert result == "Extracted text content from chart"

    # Verify call parameters
    mock_create.assert_called_once()
    args, kwargs = mock_create.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.0
    
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert "Extract all text" in content[0]["text"]
    
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"
