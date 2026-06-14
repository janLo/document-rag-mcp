# CLI Reference

The `document-rag-mcp` CLI exposes subcommands to run the RAG server, test searches, and trigger manual index scanning.

## Global Options

- `--config`, `-c`: Path to the YAML configuration file. Can also be set via the `DOCRAG_CONFIG` environment variable.
- `--chunking-model`: Override the local model name for semantic boundary splits. Can also be set via the `DOCRAG_CHUNKING_MODEL` environment variable.
- `--help`: Show the help message and exit.

## Subcommands

### `serve`
Starts the Model Context Protocol (MCP) server.

- `--transport`: Choose `stdio` (default) or `http` (SSE).
- `--host`: Bind host for SSE transport (default `127.0.0.1`).
- `--port`: Bind port for SSE transport (default `8000`).

**Example:**
```bash
document-rag-mcp serve --transport stdio
```

### `ingest`
Trigger a one-shot synchronous recursive scan and index of files in all collections (or a specific collection). This command prunes entries for deleted files.

- `--collection`, `-c`: Limit the scan to a specific collection by name.

**Example:**
```bash
document-rag-mcp ingest --collection "project-docs"
```

### `search`
Executes a semantic search against the indexed collections and prints formatted results to the terminal.

- `QUERY` (Argument, Required): The semantic search query text.
- `--collection`, `-c`: Filter search results to a specific collection by name.
- `--top-k`, `-k`: The number of nearest matches to display (default 5).

**Example:**
```bash
document-rag-mcp search "how to configure the pipeline" -k 3
```

### `collections`
Lists all collections specified in the configuration file, showing their configured directories, search file glob patterns, and the total count of indexed chunks.

**Example:**
```bash
document-rag-mcp collections
```
