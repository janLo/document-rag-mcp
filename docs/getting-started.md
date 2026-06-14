# Getting Started

Follow these steps to install, configure, and run the Document RAG MCP server.

## Installation

This project uses `uv` for python dependency management. Ensure you have `uv` installed, then synchronize the environment:

```bash
git clone https://github.com/user/document-rag-mcp.git
cd document-rag-mcp
uv sync --group dev
```

## First Configuration

Copy the example configuration file and adjust it to your directories:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to specify the folders you want to watch.

## Quick CLI Usage

You can test the ingestion and run semantic searches directly from the command line:

### 1. Ingest Documents
Run a one-shot ingestion scan to populate the vector store:
```bash
uv run document-rag-mcp ingest
```

### 2. Run a Test Search
Query the indexed collections:
```bash
uv run document-rag-mcp search "What is project antigravity?"
```

### 3. Show Collection Statistics
Check chunk counts and file listings:
```bash
uv run document-rag-mcp collections
```

## Running the MCP Server

Start the server using `stdio` transport (default) for integration with LLM clients:

```bash
uv run document-rag-mcp serve --transport stdio
```

For HTTP/SSE integration, specify the transport and bindings:
```bash
uv run document-rag-mcp serve --transport http --host 127.0.0.1 --port 8000
```
