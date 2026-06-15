import hashlib
from ..embedding.client import EmbeddingClient
from ..models import SearchResult
from ..storage.vector_store import VectorStore
from ..storage.state_store import StateStore


class SearchEngine:
    def __init__(self, vector_store: VectorStore, embedding_client: EmbeddingClient, state_store: StateStore | None = None):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.state_store = state_store

    async def search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Performs hybrid search (semantic + BM25) for a query.

        If collection_name is provided, searches only within that collection.
        If collection_name is None, searches across all existing collections and merges results.
        """
        # 1. Semantic search
        # Embed the query
        query_vector = await self.embedding_client.embed_query(query)

        if collection_name:
            # Single collection search
            semantic_results = self.vector_store.search(
                collection_name=collection_name,
                vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        else:
            # Cross-collection search: query each collection and merge
            collections = self.vector_store.list_collections()
            if not collections:
                semantic_results = []
            else:
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
                semantic_results = merged_results[:top_k]

        # 2. BM25 keyword search
        bm25_results = []
        if self.state_store:
            try:
                bm25_hits = self.state_store.search_fts(query, collection_name, top_k=top_k * 2)
                bm25_results = self._hydrate_fts_results(bm25_hits)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"FTS search failed: {e}")
                bm25_results = []

        # 3. Reciprocal Rank Fusion
        return self._rrf_merge(semantic_results, bm25_results, top_k)

    def _hydrate_fts_results(self, bm25_hits: list[tuple[str, str, float]]) -> list[SearchResult]:
        """Hydrates FTS hits (chunk_id, collection, score) by fetching details from ChromaDB."""
        if not bm25_hits:
            return []

        # Group chunk IDs by collection
        by_collection: dict[str, list[str]] = {}
        scores_by_id: dict[str, float] = {}
        for chunk_id, collection, score in bm25_hits:
            by_collection.setdefault(collection, []).append(chunk_id)
            scores_by_id[chunk_id] = score

        hydrated_map = {}
        for coll, ids in by_collection.items():
            chunks = self.vector_store.get_chunks_by_ids(coll, ids)
            for r in chunks:
                # Reconstruct chunk_id
                seed = f"{r.metadata.file_path}_{r.metadata.chunk_index}"
                c_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
                hydrated_map[c_id] = r

        # Rebuild the list in the original order of bm25_hits
        ordered_results = []
        for chunk_id, _, score in bm25_hits:
            if chunk_id in hydrated_map:
                r = hydrated_map[chunk_id]
                r.score = score
                ordered_results.append(r)
        return ordered_results

    def _rrf_merge(
        self, semantic: list[SearchResult], bm25: list[SearchResult], top_k: int, k: int = 60
    ) -> list[SearchResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(semantic, 1):
            key = r.metadata.chunk_hash
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            result_map[key] = r

        for rank, r in enumerate(bm25, 1):
            key = r.metadata.chunk_hash
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in result_map:
                result_map[key] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for key, rrf_score in ranked:
            r = result_map[key]
            r.score = rrf_score
            final_results.append(r)
        return final_results

