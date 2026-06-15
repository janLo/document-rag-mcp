from datetime import datetime, timezone
import hashlib
from pathlib import Path
from ..config import AppConfig
from ..embedding.client import EmbeddingClient
from ..storage.state_store import StateStore
from ..storage.vector_store import VectorStore
from ..vision.client import VisionClient
from .chunker import DocumentChunker
from .extractor import DocumentExtractor


def compute_file_hash(path: Path) -> str:
    """Computes the SHA-256 hash of a file's content."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        state_store: StateStore,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        vision_client: VisionClient,
    ):
        self.config = config
        self.state_store = state_store
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.vision_client = vision_client

        self.extractor = DocumentExtractor(vision_enabled=config.vision.enabled)
        self.chunker = DocumentChunker(config.chunking)

    async def ingest_file(self, file_path: Path | str, collection_name: str) -> bool:
        """Ingests a file, optimizing with file-level and chunk-level deduplication.

        Returns True if the file was ingested/updated, False if skipped because it was unchanged.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            # If the file does not exist, route to delete
            await self.delete_file(str(path), collection_name)
            return False

        file_str = str(path)
        current_hash = compute_file_hash(path)
        mtime = path.stat().st_mtime
        last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)

        # 1. File-level check
        stored_meta = self.state_store.get_file(file_str)
        if stored_meta and stored_meta["file_hash"] == current_hash:
            # File is unchanged on disk, skip indexing
            return False

        # 2. Extract pages
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Extracting file: {file_str}")
        pages = self.extractor.extract(path)

        # 3. Handle optional vision processing for scanned / text-less pages
        for page in pages:
            if page.image_bytes and self.config.vision.enabled:
                logger.info(f"Using vision processing for page {page.page_number} of {path.name}")
                ocr_text = await self.vision_client.extract_text_from_image(page.image_bytes)
                if ocr_text:
                    # Append OCR extracted text to the page text
                    page.text = (page.text + "\n" + ocr_text).strip()

        # 4. Chunk pages
        file_type = path.suffix.lower().lstrip(".")
        new_chunks = self.chunker.chunk_pages(
            pages=pages,
            file_path=file_str,
            collection_name=collection_name,
            file_hash=current_hash,
            file_type=file_type,
            last_modified=last_modified,
        )

        if not new_chunks:
            # If the file ended up with no chunks (e.g. empty file), delete any previous records
            await self.delete_file(file_str, collection_name)
            return True

        # 5. Chunk-level deduplication (reusing embeddings for unchanged paragraphs)
        old_chunks = self.state_store.get_file_chunks(file_str)
        old_chunks_by_hash = {c["chunk_hash"]: c["chunk_id"] for c in old_chunks}

        # Find which new chunks can reuse old embeddings
        reusable_ids = []
        new_chunks_to_embed = []

        for chunk in new_chunks:
            if chunk.metadata.chunk_hash in old_chunks_by_hash:
                old_id = old_chunks_by_hash[chunk.metadata.chunk_hash]
                reusable_ids.append((chunk, old_id))
            else:
                new_chunks_to_embed.append(chunk)

        # Fetch reusable embeddings from ChromaDB
        if reusable_ids:
            old_ids = [old_id for _, old_id in reusable_ids]
            # Fetch embeddings directly from vector store
            fetched_embeddings = self._get_embeddings_from_chroma(collection_name, old_ids)

            for chunk, old_id in reusable_ids:
                if old_id in fetched_embeddings:
                    chunk.embedding = fetched_embeddings[old_id]
                else:
                    # If for some reason ChromaDB lacks the vector, we must re-embed it
                    new_chunks_to_embed.append(chunk)

        # 6. Generate embeddings for new/modified chunks
        if new_chunks_to_embed:
            texts_to_embed = [
                f"[Document: {c.metadata.title or 'Unknown'} | Section: {c.metadata.section or 'Unknown'}]\n{c.text}"
                for c in new_chunks_to_embed
            ]
            vectors = await self.embedding_client.embed(texts_to_embed)
            for chunk, vector in zip(new_chunks_to_embed, vectors):
                chunk.embedding = vector

        # 7. Update Vector Store
        # Delete old file chunks first (to handle cases where total chunks changed)
        self.vector_store.delete_by_file(collection_name, file_str)
        # Upsert all chunks
        self.vector_store.upsert_chunks(collection_name, new_chunks)

        # 8. Update State Store
        chunk_states = [
            (chunk.id, chunk.metadata.chunk_hash, chunk.metadata.chunk_index)
            for chunk in new_chunks
        ]
        self.state_store.save_file_state(
            file_path=file_str,
            collection=collection_name,
            file_hash=current_hash,
            last_modified=mtime,
            chunks=chunk_states,
        )

        # 8b. Update FTS index
        fts_data = [
            (
                chunk.id,
                f"[Document: {chunk.metadata.title or 'Unknown'} | Section: {chunk.metadata.section or 'Unknown'}]\n{chunk.text}",
                collection_name
            )
            for chunk in new_chunks
        ]
        self.state_store.save_chunks_text(fts_data)

        return True

    async def delete_file(self, file_path: str, collection_name: str) -> None:
        """Removes a file and its chunks from the SQLite state store and ChromaDB index."""
        # 1. Delete from Vector Store
        self.vector_store.delete_by_file(collection_name, file_path)
        # 2. Delete from State Store (cascades chunk table entries)
        self.state_store.delete_file(file_path)

    def _get_embeddings_from_chroma(
        self, collection_name: str, ids: list[str]
    ) -> dict[str, list[float]]:
        """Queries ChromaDB to retrieve embeddings for a list of chunk IDs."""
        from ..storage.vector_store import sanitize_collection_name

        sanitized_name = sanitize_collection_name(collection_name)
        try:
            collection = self.vector_store.client.get_collection(name=sanitized_name)
            results = collection.get(ids=ids, include=["embeddings"])
            if results and results.get("embeddings") is not None:
                embeddings_dict = {}
                for i in range(len(results["ids"])):
                    c_id = results["ids"][i]
                    emb = results["embeddings"][i]
                    if emb is not None:
                        embeddings_dict[c_id] = [float(x) for x in emb]
                return embeddings_dict
        except Exception:
            pass
        return {}
