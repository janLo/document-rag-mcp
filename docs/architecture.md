# System Architecture

The Document RAG MCP server is designed as a modular, lightweight, and performant pipeline.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Server (FastMCP)                    │
│  Transport: stdio | streamable-http                             │
│                                                                 │
│  Tools:     search, list_collections, get_document_content,     │
│             get_document_original, ingest_now                   │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
     ┌─────▼──────┐              ┌────────▼────────┐
     │  Ingestion  │              │   Search        │
     │  Pipeline   │              │   Engine        │
     │             │              │                 │
     │  extract →  │              │  embed query →  │
     │  chunk →    │              │  vector query → │
     │  hash →     │              │  merge & rank   │
     │  upsert     │              └────────┬────────┘
     └─────┬──────┘                        │
           │                               │
     ┌─────▼──────┐                  ┌─────▼──────┐
     │State Store │                  │Vector Store│
     │  (SQLite)  │                  │ (ChromaDB) │
     └────────────┘                  └────────────┘
```

## Modular Description

- **Extraction (`extractor.py`)**: Responsible for reading plain text, parsing frontmatter/headings in markdown, and extraction from PDFs via PyMuPDF. It uses font sizes and tags in the PDF layout to detect headers.
- **Chunking (`chunker.py`)**: Uses `chonkie` to partition text into tokens. Markdown is recursively split at section headers and paragraph shifts, while TXT/PDF is semantically chunked.
- **Incremental Pipeline (`pipeline.py`)**: Checks file-level hashes (SHA-256) and stores them in SQLite. If a file is modified, it computes new chunk hashes, maps existing chunk vectors from ChromaDB, and only requests embeddings for new/modified chunks, saving API tokens.
- **Search Engine (`engine.py`)**: Performs semantic searches by generating a vector representation of the query and querying ChromaDB. For multi-collection queries, it merges and ranks results using ascending order of L2 distance.
- **Security Boundaries**: Path operations inside the MCP tools (`get_document_content`, `get_document_original`) validate that the target files reside within one of the collection directories before performing file system reads, protecting the server host from arbitrary path traversal attacks.
