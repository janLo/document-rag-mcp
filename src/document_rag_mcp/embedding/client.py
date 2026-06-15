import time
import logging
from openai import AsyncOpenAI
from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=120.0,  # Increase timeout to handle slow local servers
        )
        self.model = config.model
        self.dimensions = config.dimensions
        self.batch_size = config.batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            start_time = time.perf_counter()
            try:
                # Try sending with dimensions parameter
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimensions,
                )
            except Exception:
                # Fallback for APIs that don't support the dimensions parameter
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
            end_time = time.perf_counter()
            duration = end_time - start_time
            total_chars = sum(len(t) for t in batch)
            logger.debug(
                f"Embedding batch of {len(batch)} texts ({total_chars} total chars) "
                f"took {duration:.4f}s"
            )

            # Extract embeddings in correct order
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generates embedding for a single search query."""
        results = await self.embed([query])
        if not results:
            raise ValueError("Failed to generate embedding for query.")
        return results[0]
