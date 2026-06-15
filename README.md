# document-rag-mcp

A high-performance Model Context Protocol (MCP) server for local document search and extraction. It recursively scans and watches configured directories for `.txt`, `.md`, and `.pdf` files, indexes their content, and exposes them as tools for LLMs.

📖 **Full Documentation**: [https://janlo.github.io/document-rag-mcp/](https://janlo.github.io/document-rag-mcp/)

## Key Features

- **Hybrid Search (Semantic + BM25)**: Blends dense semantic vector search (ChromaDB) with sparse keyword search (SQLite FTS5) using Reciprocal Rank Fusion (RRF) for optimal retrieval.
- **Section-Grain Chunking**: Text from all pages is unified and chunked as a single stream, then mapped back to its primary page and section via character offsets, preventing artificial boundaries at page borders.
- **TOC-Aware Extractor**: Extracts PDF headings using the document's own Table of Contents (TOC), falling back to typography-aware layout detection if TOC is missing.
- **Incremental Indexing**: Uses content hashing (SHA-256) at both the file and chunk levels. Files that have not changed are skipped completely, and modified files only re-embed chunks that actually changed.
- **Auto-Pruning**: Automatically detects when files are deleted from the disk and prunes them from the index.
- **Multimodal OCR**: Detects scanned or text-less PDF pages and routes them through an optional vision-capable LLM.
- **MCP Native**: Exposes tools for hybrid search, collection statistics, metadata analysis, and full document text/binary content retrieval.

## Quick Start

### 1. Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
git clone https://github.com/janlo/document-rag-mcp.git
cd document-rag-mcp
uv sync --group dev
```

### 2. Configuration
Copy the example configuration:
```bash
cp config.example.yaml config.yaml
```
And edit `config.yaml` to specify the folders you want to watch.

### 3. CLI Commands
- **Ingest**: `uv run document-rag-mcp ingest`
- **Search**: `uv run document-rag-mcp search "your query"`
- **Start MCP Server**: `uv run document-rag-mcp serve`

