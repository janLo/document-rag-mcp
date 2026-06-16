from datetime import datetime
from pathlib import Path
import re
import chromadb
from ..models import ChunkMetadata, DocumentChunk, SearchResult


def sanitize_collection_name(name: str) -> str:
    """Sanitizes a collection name to comply with ChromaDB naming rules.

    Rules:
    - 3-63 characters long
    - Starts and ends with an alphanumeric character
    - Contains only alphanumeric characters, underscores, hyphens, or dots
    - No consecutive dots
    """
    # Replace any character that isn't alphanumeric, underscore, hyphen, or dot with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

    # Remove consecutive dots
    sanitized = re.sub(r"\.\.+", ".", sanitized)

    # Ensure length is between 3 and 63 characters
    if len(sanitized) < 3:
        sanitized = sanitized.ljust(3, "0")
    elif len(sanitized) > 63:
        sanitized = sanitized[:63]

    # Ensure it starts with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = "a" + sanitized[1:]

    # Ensure it ends with alphanumeric
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + "a"

    return sanitized


class VectorStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.client = chromadb.PersistentClient(path=str(data_dir / "chroma"))

    def ensure_collection(self, name: str) -> None:
        """Ensures a collection exists under the sanitized name."""
        sanitized = sanitize_collection_name(name)
        self.client.get_or_create_collection(name=sanitized)

    def upsert_chunks(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Upserts a list of document chunks into ChromaDB."""
        if not chunks:
            return

        sanitized = sanitize_collection_name(collection_name)
        collection = self.client.get_or_create_collection(name=sanitized)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} does not have an embedding.")

            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)

            # Chroma metadata values must be str, int, float, or bool.
            meta_dict = chunk.metadata.model_dump()
            serialized_meta = {}
            for k, v in meta_dict.items():
                if v is None:
                    continue
                if isinstance(v, datetime):
                    serialized_meta[k] = v.isoformat()
                elif isinstance(v, Path):
                    serialized_meta[k] = str(v)
                elif isinstance(v, (str, int, float, bool)):
                    serialized_meta[k] = v
                else:
                    serialized_meta[k] = str(v)

            metadatas.append(serialized_meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_file(self, collection_name: str, file_path: str) -> None:
        """Deletes all chunks belonging to a specific file path from a collection."""
        sanitized = sanitize_collection_name(collection_name)
        try:
            collection = self.client.get_collection(name=sanitized)
            collection.delete(where={"file_path": file_path})
        except Exception:
            # Collection might not exist yet, which is fine
            pass

    def search(
        self,
        collection_name: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Searches for chunks in a collection that are closest to the query vector."""
        sanitized = sanitize_collection_name(collection_name)
        try:
            collection = self.client.get_collection(name=sanitized)
        except Exception:
            return []

        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
        )

        search_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                meta_dict = metas[i]

                # Deserialize datetime fields
                if "last_modified" in meta_dict and isinstance(meta_dict["last_modified"], str):
                    meta_dict["last_modified"] = datetime.fromisoformat(meta_dict["last_modified"])
                if "ingested_at" in meta_dict and isinstance(meta_dict["ingested_at"], str):
                    meta_dict["ingested_at"] = datetime.fromisoformat(meta_dict["ingested_at"])

                metadata = ChunkMetadata(**meta_dict)
                score = distances[i] if i < len(distances) else 0.0

                search_results.append(
                    SearchResult(
                        text=docs[i],
                        score=score,
                        metadata=metadata,
                    )
                )

        return search_results

    def get_chunks_by_ids(self, collection_name: str, ids: list[str]) -> list[SearchResult]:
        """Retrieves specific chunks by their IDs from ChromaDB and returns them as SearchResult objects."""
        if not ids:
            return []
        sanitized = sanitize_collection_name(collection_name)
        try:
            collection = self.client.get_collection(name=sanitized)
            results = collection.get(ids=ids, include=["documents", "metadatas"])
            
            search_results = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"]
                metas = results["metadatas"]
                
                for i in range(len(docs)):
                    meta_dict = metas[i]
                    
                    # Deserialize datetime fields
                    if "last_modified" in meta_dict and isinstance(meta_dict["last_modified"], str):
                        meta_dict["last_modified"] = datetime.fromisoformat(meta_dict["last_modified"])
                    if "ingested_at" in meta_dict and isinstance(meta_dict["ingested_at"], str):
                        meta_dict["ingested_at"] = datetime.fromisoformat(meta_dict["ingested_at"])
                    
                    metadata = ChunkMetadata(**meta_dict)
                    
                    search_results.append(
                        SearchResult(
                            text=docs[i],
                            score=0.0,  # Score is not meaningful for direct ID fetch
                            metadata=metadata,
                        )
                    )
                return search_results
        except Exception:
            pass
        return []

    def list_collections(self) -> list[str]:
        """Returns the names of all collections in ChromaDB."""
        return [c.name for c in self.client.list_collections()]

    def collection_stats(self, name: str) -> dict[str, int]:
        """Returns statistics for a collection (like document count)."""
        sanitized = sanitize_collection_name(name)
        try:
            collection = self.client.get_collection(name=sanitized)
            return {"count": collection.count()}
        except Exception:
            return {"count": 0}
