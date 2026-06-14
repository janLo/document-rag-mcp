from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner
from document_rag_mcp.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Start the Model Context Protocol (MCP) server" in result.output
    assert "serve" in result.output
    assert "search" in result.output
    assert "ingest" in result.output
    assert "collections" in result.output


def test_cli_collections():
    runner = CliRunner()
    
    # Mock server config and stats
    mock_coll = MagicMock()
    mock_coll.name = "test-coll"
    mock_coll.paths = ["/docs"]
    mock_coll.file_patterns = ["*.md"]
    
    with patch("document_rag_mcp.server.config") as mock_config, \
         patch("document_rag_mcp.server.vector_store") as mock_vector_store:
         
        mock_config.collections = [mock_coll]
        mock_vector_store.collection_stats.return_value = {"count": 42}
        
        result = runner.invoke(main, ["collections"])
        assert result.exit_code == 0
        assert "test-coll" in result.output
        assert "42" in result.output
        assert "*.md" in result.output


def test_cli_search():
    runner = CliRunner()
    
    # Mock search results
    mock_search = AsyncMock(return_value=[])
    
    with patch("document_rag_mcp.server.search_engine.search", new=mock_search):
        result = runner.invoke(main, ["search", "my query", "-c", "my-coll", "-k", "3"])
        
        assert result.exit_code == 0
        assert "Searching for: 'my query'" in result.output
        assert "No matching documents found." in result.output
        mock_search.assert_called_once_with(query="my query", collection_name="my-coll", top_k=3)


def test_cli_ingest():
    runner = CliRunner()
    
    # Mock ingest_now call
    mock_ingest = AsyncMock(return_value="Ingestion completed. Ingested/Updated: 5, Skipped: 0.")
    
    with patch("document_rag_mcp.server.ingest_now", new=mock_ingest):
        result = runner.invoke(main, ["ingest", "-c", "my-coll"])
        
        assert result.exit_code == 0
        assert "Starting one-shot ingestion scan" in result.output
        assert "Ingested/Updated: 5" in result.output
        mock_ingest.assert_called_once_with(collection="my-coll")
