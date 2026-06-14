from datetime import datetime
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    file_path: str
    file_name: str
    collection: str
    file_hash: str                    # SHA-256 of file content
    chunk_hash: str                   # SHA-256 of chunk text
    chunk_index: int
    total_chunks: int
    title: str | None = None          # extracted from frontmatter/h1
    section: str | None = None        # nearest heading
    file_type: str                    # "txt", "md", "pdf"
    page_number: int | None = None    # PDF only (1-indexed)
    last_modified: datetime
    ingested_at: datetime
    vision_processed: bool = False


class DocumentChunk(BaseModel):
    id: str                           # deterministic: sha256(file_path + chunk_index)
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata
