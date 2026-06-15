from datetime import datetime, timezone
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from .config import CollectionConfig, load_config
from .embedding.client import EmbeddingClient
from .ingestion.pipeline import IngestionPipeline
from .search.engine import SearchEngine
from .storage.state_store import StateStore
from .storage.vector_store import VectorStore
from .vision.client import VisionClient

# 1. Global context initialization from environment variable DOCRAG_CONFIG
config = load_config()
state_store = StateStore(config.storage.data_dir)
vector_store = VectorStore(config.storage.data_dir)
embedding_client = EmbeddingClient(config.embedding)
vision_client = VisionClient(config.vision)
pipeline = IngestionPipeline(
    config=config,
    state_store=state_store,
    vector_store=vector_store,
    embedding_client=embedding_client,
    vision_client=vision_client,
)
search_engine = SearchEngine(vector_store, embedding_client, state_store)

# 2. FastMCP server definition
mcp = FastMCP("document-rag-mcp")


def is_safe_path(file_path: str, collections: list[CollectionConfig]) -> bool:
    """Security helper to validate that a file path resides within a configured collection folder.

    Prevents directory traversal attacks.
    """
    target_path = Path(file_path).resolve()
    for coll in collections:
        for folder in coll.paths:
            resolved_folder = folder.resolve()
            if target_path == resolved_folder or target_path.is_relative_to(resolved_folder):
                return True
    return False


@mcp.tool()
async def search(query: str, collection: str | None = None, top_k: int = 10) -> str:
    """Search for document chunks semantically matching the query.

    Args:
        query: The semantic search query.
        collection: Optional collection name to filter results by.
        top_k: Number of search results to return (default 10).
    """
    results = await search_engine.search(query, collection, top_k)
    if not results:
        return "No matching documents found."

    output = []
    for i, r in enumerate(results):
        output.append(
            f"[{i+1}] File: {r.metadata.file_path}\n"
            f"Collection: {r.metadata.collection}\n"
            f"Title: {r.metadata.title or 'N/A'}\n"
            f"Section: {r.metadata.section or 'N/A'}\n"
            f"Page: {r.metadata.page_number or 1}\n"
            f"Similarity Score (L2 Distance): {r.score:.4f}\n"
            f"Content:\n{r.text.strip()}\n"
            f"---"
        )
    return "\n\n".join(output)


@mcp.tool()
async def list_collections() -> str:
    """List all configured collections and their indexing status."""
    if not config.collections:
        return "No collections configured."

    output = []
    for coll in config.collections:
        stats = vector_store.collection_stats(coll.name)
        paths_str = ", ".join(str(p) for p in coll.paths)
        output.append(
            f"Collection: {coll.name}\n"
            f"Paths: {paths_str}\n"
            f"File Patterns: {', '.join(coll.file_patterns)}\n"
            f"Indexed Chunks: {stats['count']}\n"
            f"---"
        )
    return "\n\n".join(output)


@mcp.tool()
async def get_document_info(file_path: str) -> str:
    """Get metadata about an ingested document.

    Args:
        file_path: The absolute path to the document file.
    """
    file_meta = state_store.get_file(file_path)
    if not file_meta:
        return f"Document '{file_path}' is not indexed or tracked."

    last_mod = datetime.fromtimestamp(file_meta["last_modified"], tz=timezone.utc).isoformat()
    return (
        f"File Path: {file_meta['file_path']}\n"
        f"Collection: {file_meta['collection']}\n"
        f"Content Hash: {file_meta['file_hash']}\n"
        f"Chunk Count: {file_meta['chunk_count']}\n"
        f"Last Modified: {last_mod}\n"
        f"Ingested At: {file_meta['ingested_at']}"
    )


@mcp.tool()
async def get_document_content(file_path: str) -> str:
    """Retrieve the full text content of a document directly from the disk.

    Args:
        file_path: The absolute path to the document file.
    """
    if not is_safe_path(file_path, config.collections):
        return "Access Denied: Path is not inside any configured collection folders."

    path = Path(file_path)
    if not path.exists():
        return f"File '{file_path}' does not exist on disk."

    from .ingestion.extractor import DocumentExtractor

    extractor = DocumentExtractor(vision_enabled=False)  # CPU text extraction only
    try:
        pages = extractor.extract(path)
        return "\n\n".join(page.text for page in pages)
    except Exception as e:
        return f"Error extracting text from {file_path}: {e}"


@mcp.tool()
async def get_document_original(file_path: str) -> bytes:
    """Retrieve the raw binary bytes of a document from the disk (e.g. PDF file).

    Args:
        file_path: The absolute path to the document file.
    """
    if not is_safe_path(file_path, config.collections):
        raise PermissionError("Access Denied: Path is not inside any configured collection folders.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    return path.read_bytes()


@mcp.tool()
async def ingest_now(collection: str | None = None) -> str:
    """Force an immediate recursive scan and indexing of folders.

    Prunes deleted files.

    Args:
        collection: Optional collection name to ingest (default scans all collections).
    """
    from .ingestion.scanner import scan_collection_files

    colls_to_scan = config.collections
    if collection:
        colls_to_scan = [c for c in config.collections if c.name == collection]
        if not colls_to_scan:
            return f"Collection '{collection}' not found in configuration."

    total_ingested = 0
    total_skipped = 0

    for coll in colls_to_scan:
        # 1. Scan files on disk
        files_on_disk = scan_collection_files(coll)
        disk_paths = {str(p.resolve()) for p in files_on_disk}

        # 2. Ingest / Update
        for path in files_on_disk:
            try:
                updated = await pipeline.ingest_file(path, coll.name)
                if updated:
                    total_ingested += 1
                else:
                    total_skipped += 1
            except Exception as e:
                print(f"Error ingesting {path}: {e}")

        # 3. Prune deleted files
        stored_files = state_store.list_collection_files(coll.name)
        for f in stored_files:
            f_path = f["file_path"]
            if f_path not in disk_paths:
                await pipeline.delete_file(f_path, coll.name)

    return f"Ingestion completed. Ingested/Updated: {total_ingested}, Skipped: {total_skipped}."


# 3. Resources definition
@mcp.resource("collections://list")
async def collections_resource() -> str:
    """Resource returning metadata list of all configured collections."""
    data = []
    for coll in config.collections:
        stats = vector_store.collection_stats(coll.name)
        data.append(
            {
                "name": coll.name,
                "paths": [str(p) for p in coll.paths],
                "file_patterns": coll.file_patterns,
                "indexed_chunks": stats["count"],
            }
        )
    return json.dumps(data, indent=2)


@mcp.resource("collection://{name}/info")
async def collection_info_resource(name: str) -> str:
    """Resource returning detailed metadata info of a specific collection."""
    coll = next((c for c in config.collections if c.name == name), None)
    if not coll:
        return json.dumps({"error": f"Collection '{name}' not found"})

    stats = vector_store.collection_stats(name)
    files = state_store.list_collection_files(name)

    return json.dumps(
        {
            "name": coll.name,
            "paths": [str(p) for p in coll.paths],
            "file_patterns": coll.file_patterns,
            "indexed_chunks": stats["count"],
            "files": [
                {
                    "file_path": f["file_path"],
                    "file_hash": f["file_hash"],
                    "last_modified": datetime.fromtimestamp(
                        f["last_modified"], tz=timezone.utc
                    ).isoformat(),
                }
                for f in files
            ],
        },
        indent=2,
    )


@mcp.resource("collection://{name}/stats")
async def collection_stats_resource(name: str) -> str:
    """Resource returning count statistics of a specific collection."""
    stats = vector_store.collection_stats(name)
    return json.dumps(stats, indent=2)
