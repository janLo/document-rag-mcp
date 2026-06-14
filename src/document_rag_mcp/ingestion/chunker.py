from datetime import datetime, timezone
import hashlib
import os
from ..config import ChunkingConfig
from ..models import ChunkMetadata, DocumentChunk
from .extractor import ExtractedPage


class DocumentChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._init_chunkers()

    def _init_chunkers(self) -> None:
        import chonkie

        # 1. Initialize semantic chunker (for TXT / PDF)
        try:
            self.semantic_chunker = chonkie.SemanticChunker(
                embedding_model=self.config.local_model,
                chunk_size=self.config.max_chunk_size,
                threshold=self.config.similarity_threshold,
            )
        except Exception as e:
            # Fallback to SentenceChunker/TokenChunker if sentence-transformers is missing or fails
            print(
                f"Warning: Failed to load SemanticChunker with '{self.config.local_model}': {e}"
            )
            print("Falling back to TokenChunker for semantic text.")
            self.semantic_chunker = chonkie.TokenChunker(chunk_size=self.config.max_chunk_size)

        # 2. Initialize recursive chunker for Markdown (uses split rules rather than semantic embeddings)
        self.markdown_chunker = chonkie.RecursiveChunker(
            chunk_size=self.config.max_chunk_size,
        )

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        file_path: str,
        collection_name: str,
        file_hash: str,
        file_type: str,
        last_modified: datetime,
    ) -> list[DocumentChunk]:
        """Transforms a list of ExtractedPages into a flat list of metadata-rich DocumentChunks."""
        file_name = os.path.basename(file_path)

        # Resolve document title once for the entire document
        document_title = None
        if pages and pages[0].metadata:
            document_title = pages[0].metadata.get("title")
        if not document_title and pages and pages[0].headings:
            for heading_text, level in pages[0].headings:
                if level == 1:
                    document_title = heading_text
                    break
        if not document_title:
            document_title = os.path.splitext(file_name)[0]

        raw_chunks: list[tuple[str, int | None, list[tuple[str, int]]]] = []

        # Generate chunks per page
        for page in pages:
            text_to_chunk = page.text.strip()
            if not text_to_chunk:
                continue

            if file_type == "md":
                chunks_out = self.markdown_chunker.chunk(text_to_chunk)
            else:
                chunks_out = self.semantic_chunker.chunk(text_to_chunk)

            for c in chunks_out:
                raw_chunks.append((c.text, page.page_number, page.headings))

        total_chunks = len(raw_chunks)
        document_chunks: list[DocumentChunk] = []

        for idx, (chunk_text, page_num, headings) in enumerate(raw_chunks):
            # Resolve section / heading
            section = None
            if headings:
                # Check headings in reverse order (closest heading before/within the chunk)
                for heading_text, _ in reversed(headings):
                    if heading_text in chunk_text:
                        section = heading_text
                        break
                if not section:
                    section = headings[0][0]

            # Compute chunk-level hash
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            # Deterministic chunk ID
            chunk_id_seed = f"{file_path}_{idx}"
            chunk_id = hashlib.sha256(chunk_id_seed.encode("utf-8")).hexdigest()

            metadata = ChunkMetadata(
                file_path=file_path,
                file_name=file_name,
                collection=collection_name,
                file_hash=file_hash,
                chunk_hash=chunk_hash,
                chunk_index=idx,
                total_chunks=total_chunks,
                title=document_title,
                section=section,
                file_type=file_type,
                page_number=page_num,
                last_modified=last_modified,
                ingested_at=datetime.now(timezone.utc),
                vision_processed=False,
            )

            document_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                    embedding=None,
                )
            )

        return document_chunks
