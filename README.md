# document-rag-mcp

A RAG MCP (Model Context Protocol) server. It recursively scans and watches configured directories for `.txt`, `.md`, and `.pdf` files, semantically chunks them, computes embeddings using an OpenAI-compatible API, and stores them in an embedded ChromaDB instance.

It exposes tools to search documents and retrieve original content/metadata over the MCP protocol.
