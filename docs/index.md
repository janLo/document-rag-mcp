# Document RAG MCP Server

Welcome to the **document-rag-mcp** server documentation!

This Model Context Protocol (MCP) server enables semantic search and content extraction over text, Markdown, and PDF documents. It works fully in-process without requiring external database servers, using ChromaDB as the vector store and SQLite as the ingestion state tracker.

## Key Features

- **Recursive Folder Scanning**: Configured folders are recursively scanned on startup and then monitored in real-time via inotify (`watchfiles`).
- **Incremental Indexing**: Uses content hashing (SHA-256) at both the file and chunk levels. Files that have not changed are skipped completely on startup, and modified files only re-embed chunks that actually changed.
- **Auto-Pruning**: Automatically detects when files are deleted from the disk (both while running and when offline) and prunes them from the index.
- **Multimodal PDF Processing**: Detects scanned or text-less PDF pages and routes them through an optional vision-capable LLM to extract text.
- **MCP Native**: Exposes tools for semantic search, collection stats, metadata analysis, and full document text/binary content retrieval.
- **Secure Boundaries**: Strictly validates all path parameters against configured collections folders to protect against directory traversal attacks.
