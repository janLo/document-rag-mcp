import asyncio
from document_rag_mcp.storage.state_store import StateStore
from document_rag_mcp.storage.vector_store import VectorStore
from document_rag_mcp.embedding.client import EmbeddingClient
from document_rag_mcp.search.engine import SearchEngine
from document_rag_mcp.config import load_config

async def main():
    config = load_config("config.yaml")
    state_store = StateStore(config.storage.data_dir)
    vector_store = VectorStore(config.storage.data_dir)
    embed_client = EmbeddingClient(config.embedding)
    
    engine = SearchEngine(vector_store, embed_client, state_store=state_store)
    
    query = "Personalampel"
    query_vector = await embed_client.embed_query(query)
    
    sem_results = engine.vector_store.search(collection_name="notes-kita", vector=query_vector, top_k=20)
    for res in sem_results:
        print(f"Distance: {res.score:.4f} | File: {res.metadata.file_name}")

if __name__ == "__main__":
    asyncio.run(main())
