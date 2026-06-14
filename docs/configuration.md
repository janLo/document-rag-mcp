# Configuration Reference

The server loads configuration from a YAML file. You can specify the config file path using the `-c`/`--config` CLI flag or the `DOCRAG_CONFIG` environment variable.

## Environment Variables Override

All configuration fields can be overridden using environment variables prefixed with `DOCRAG_` and using a double underscore `__` for nesting.

**Examples:**
- Override storage directory: `DOCRAG_STORAGE__DATA_DIR="/tmp/data"`
- Override embedding model: `DOCRAG_EMBEDDING__MODEL="text-embedding-3-small"`
- Override vision enabled: `DOCRAG_VISION__ENABLED="true"`

## Detailed Configuration Options

### Collections Settings
Configure folders and matching patterns:
- `name`: Unique name of the collection (conform to alphanumeric rules).
- `paths`: Absolute paths to folders/files to index.
- `file_patterns`: Glob patterns, e.g. `["*.txt", "*.md", "*.pdf"]`.

### Embedding Settings
Configure the OpenAI-compatible embedding API:
- `base_url`: Target endpoint (e.g. `http://localhost:8080/v1` for lemonade).
- `api_key`: Authorization API Key (use "unused" if endpoint does not require one).
- `model`: Embedding model name.
- `dimensions`: Vector dimensions size (e.g., 768 for Gemma).
- `batch_size`: Maximum texts sent in a single batch request.

### Local Chunking Settings
Configure document splitting limits:
- `max_chunk_size`: Maximum number of tokens per chunk.
- `similarity_threshold`: Threshold for semantic boundary splits.
- `local_model`: Local model name (e.g. `all-MiniLM-L6-v2`) used for chunking boundaries.

### Vision Settings (Optional)
Configure scanned PDF page-to-image extraction:
- `enabled`: Set `true` to enable multimodal OCR fallback.
- `base_url`: OpenAI-compatible vision completion endpoint.
- `model`: Multimodal model name (e.g. `gpt-4o`).
