import asyncio
from document_rag_mcp.storage.state_store import StateStore
from document_rag_mcp.storage.vector_store import VectorStore
from document_rag_mcp.embedding.client import EmbeddingClient
from document_rag_mcp.search.engine import SearchEngine
from document_rag_mcp.config import load_config
import hashlib

async def main():
    config = load_config("config.yaml")
    state_store = StateStore(config.storage.data_dir)
    vector_store = VectorStore(config.storage.data_dir)
    embed_client = EmbeddingClient(config.embedding)
    
    engine = SearchEngine(vector_store, embed_client, state_store=state_store)
    
    query = "Personalampel"
    print(f"--- FTS Hits for '{query}' ---")
    fts_results = engine.state_store.search_fts(query, top_k=20)
    for hit in fts_results:
        print(f"BM25 Hit: Chunk {hit[0]}, Coll: {hit[1]}, Score: {hit[2]}")
        
    print("\n--- Hydrating ---")
    by_collection = {}
    for hit in fts_results:
        by_collection.setdefault(hit[1], []).append(hit[0])
        
    hydrated_map = {}
    for coll, ids in by_collection.items():
        chunks = engine.vector_store.get_chunks_by_ids(coll, ids)
        print(f"Collection {coll} has {len(chunks)} chunks returned from VectorStore for {len(ids)} ids")
        for r in chunks:
            seed = f"{r.metadata.file_path}_{r.metadata.chunk_index}"
            c_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            hydrated_map[c_id] = r
            print(f"  VectorStore Chunk: {r.metadata.file_path} idx={r.metadata.chunk_index} -> c_id={c_id}")
            if c_id not in ids:
                print(f"    WARNING: c_id {c_id} is not in requested ids!")
                
    for hit in fts_results:
        c_id = hit[0]
        if c_id in hydrated_map:
            print(f"SUCCESS: hydrated {c_id} -> {hydrated_map[c_id].metadata.file_name}")
        else:
            print(f"FAILED: could not hydrate {c_id}")

if __name__ == "__main__":
    asyncio.run(main())
