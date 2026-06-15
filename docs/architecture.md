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
     │  chunk →    │              │  vector & FTS   │
     │  hash →     │              │  queries →      │
     │  upsert &   │              │  RRF merge &    │
     │  FTS index  │              │  rank           │
     └─────┬──────┘              └───────┬──┬──────┘
           │                             │  │
           │        ┌────────────────────┘  │
           ▼        ▼                       ▼
     ┌────────────────┐              ┌────────────┐
     │  State Store   │              │Vector Store│
     │ (SQLite, FTS5) │              │ (ChromaDB) │
     └────────────────┘              └────────────┘
```

## Modular Description

- **Extraction (`extractor.py`)**: Responsible for reading plain text, parsing frontmatter/headings in markdown, and extraction from PDFs via PyMuPDF. It primarily leverages the PDF's internal Table of Contents (TOC) to resolve page-level headings, falling back to a typography-aware layout detection (analyzing font sizes, weights, and layout positioning) when no TOC headings are available.
- **Chunking (`chunker.py`)**: Performs section-grain chunking. Instead of chunking documents page-by-page, the text from all pages is unified into a single document-wide text stream. The chunker (recursive splitting for Markdown, semantic chunking using `chonkie` for PDFs/TXTs) is run once over the entire text. Resulting chunks are then mapped back to their primary page and nearest preceding heading via character offset lookups. This avoids artificial chunk boundaries at page boundaries.
- **Incremental Pipeline (`pipeline.py`)**: Checks file-level hashes (SHA-256) and stores them in SQLite. If a file is modified, it computes new chunk hashes, maps existing chunk vectors from ChromaDB, and only requests embeddings for new/modified chunks, saving API tokens. It also updates the FTS5 text index in the state store.
- **Search Engine (`engine.py`)**: Implements hybrid search combining dense semantic search and sparse keyword search. The engine generates query embeddings to search ChromaDB for semantic matches, and executes a full-text search (FTS5 BM25) query against the state store. The resulting ranked lists are merged and re-ranked using Reciprocal Rank Fusion (RRF).
- **State Store (`state_store.py`)**: A SQLite database that tracks indexed document metadata and chunk hashes to enable incremental indexing and auto-pruning. It also maintains a virtual table (`chunks_fts`) using the SQLite FTS5 extension (`unicode61` tokenizer) to index chunk text and perform keyword-based BM25 search.
- **Security Boundaries**: Path operations inside the MCP tools (`get_document_content`, `get_document_original`) validate that the target files reside within one of the collection directories before performing file system reads, protecting the server host from arbitrary path traversal attacks.
