from ..embedding.client import EmbeddingClient
from ..models import SearchResult
from ..storage.vector_store import VectorStore


class SearchEngine:
    def __init__(self, vector_store: VectorStore, embedding_client: EmbeddingClient):
        self.vector_store = vector_store
        self.embedding_client = embedding_client

    async def search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Performs semantic search for a query.

        If collection_name is provided, searches only within that collection.
        If collection_name is None, searches across all existing collections and merges results.
        """
        # Embed the query
        query_vector = await self.embedding_client.embed_query(query)

        if collection_name:
            # Single collection search
            return self.vector_store.search(
                collection_name=collection_name,
                vector=query_vector,
                top_k=top_k,
                filters=filters,
            )

        # Cross-collection search: query each collection and merge
        collections = self.vector_store.list_collections()
        if not collections:
            return []

        merged_results: list[SearchResult] = []
        for coll in collections:
            results = self.vector_store.search(
                collection_name=coll,
                vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
            merged_results.extend(results)

        # Sort by similarity score (Chroma default is L2 distance, where smaller is more similar)
        merged_results.sort(key=lambda r: r.score)

        return merged_results[:top_k]
