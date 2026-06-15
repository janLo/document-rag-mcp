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

        # 1. Initialize semantic chunker
        try:
            self.semantic_chunker = chonkie.SemanticChunker(
                embedding_model=self.config.local_model,
                chunk_size=self.config.max_chunk_size,
                threshold=self.config.similarity_threshold,
            )
        except Exception as e:
            # Fallback to TokenChunker if sentence-transformers is missing or fails
            print(
                f"Warning: Failed to load SemanticChunker with '{self.config.local_model}': {e}"
            )
            print("Falling back to TokenChunker for semantic text.")
            overlap = min(
                self.config.chunk_overlap,
                max(0, self.config.max_chunk_size - 1)
            )
            self.semantic_chunker = chonkie.TokenChunker(
                chunk_size=self.config.max_chunk_size,
                chunk_overlap=overlap,
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
        """Transforms a list of ExtractedPages into a flat list of metadata-rich DocumentChunks by merging pages and chunking once."""
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

        # 1. Build unified text + page/heading offset maps
        parts, page_ranges, heading_offsets = [], [], []
        offset = 0
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            start = offset
            parts.append(text)
            offset += len(text)
            page_ranges.append((start, offset, page.page_number))
            # Record heading positions within the merged text
            for heading_text, level in page.headings:
                pos = text.find(heading_text)
                if pos >= 0:
                    heading_offsets.append((start + pos, heading_text, level))
            parts.append("\n\n")
            offset += 2

        full_text = "".join(parts).rstrip()
        if not full_text:
            return []

        # 2. Section-Bound Chunking
        # Ensure heading offsets are sorted by position
        heading_offsets.sort(key=lambda x: x[0])

        # Define sections using the heading offsets
        sections = []
        if not heading_offsets:
            sections.append((0, len(full_text), None))
        else:
            # First section before any heading (if any)
            if heading_offsets[0][0] > 0:
                sections.append((0, heading_offsets[0][0], None))
            for i in range(len(heading_offsets)):
                start = heading_offsets[i][0]
                end = heading_offsets[i+1][0] if i + 1 < len(heading_offsets) else len(full_text)
                heading_text = heading_offsets[i][1]
                sections.append((start, end, heading_text))

            # Clean up the first section if it contains no alphanumeric characters (e.g. just '# ')
            if len(sections) > 1 and sections[0][2] is None:
                first_start, first_end, _ = sections[0]
                if not any(c.isalnum() for c in full_text[first_start:first_end]):
                    # Merge it into the next section (the first heading section)
                    second_start, second_end, second_name = sections[1]
                    sections[1] = (first_start, second_end, second_name)
                    sections.pop(0)

        chunks_out = []
        from chonkie import Chunk
        for start, end, section_name in sections:
            section_text = full_text[start:end]
            if not section_text.strip():
                continue

            num_tokens = self.semantic_chunker.tokenizer.count_tokens(section_text)
            if num_tokens <= self.config.max_chunk_size:
                # Keep it as a single chunk
                chunks_out.append(
                    Chunk(
                        text=section_text,
                        start_index=start,
                        end_index=end,
                        token_count=num_tokens,
                    )
                )
            else:
                # Chunk it semantically (or with TokenChunker if that's the fallback)
                sub_chunks = self.semantic_chunker.chunk(section_text)
                for sc in sub_chunks:
                    # Adjust start_index and end_index to be relative to full_text
                    sc.start_index += start
                    sc.end_index += start
                    chunks_out.append(sc)

        total_chunks = len(chunks_out)
        document_chunks: list[DocumentChunk] = []

        for idx, chunk in enumerate(chunks_out):
            chunk_text = chunk.text
            # 3. Map each chunk to page + section via offset lookup
            page_num = self._find_primary_page(chunk.start_index, chunk.end_index, page_ranges)
            section = self._find_section_at_offset(chunk.start_index, chunk.end_index, heading_offsets)

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

    def _find_primary_page(self, start: int, end: int, page_ranges: list[tuple[int, int, int]]) -> int | None:
        """Page where the majority of the chunk text lives."""
        best_page, best_overlap = None, 0
        for p_start, p_end, page_num in page_ranges:
            overlap = max(0, min(end, p_end) - max(start, p_start))
            if overlap > best_overlap:
                best_overlap, best_page = overlap, page_num
        if best_page is None and page_ranges:
            best_page = page_ranges[0][2]
        return best_page

    def _find_section_at_offset(self, chunk_start: int, chunk_end: int, heading_offsets: list[tuple[int, str, int]]) -> str | None:
        """Last heading that precedes the chunk start, or falls back to the first heading in/near the chunk."""
        if not heading_offsets:
            return None

        section = None
        # 1. Try to find the last heading that starts before or at chunk_start
        for offset, text, level in heading_offsets:
            if offset <= chunk_start:
                section = text
            else:
                break

        if section is not None:
            return section

        # 2. If chunk starts before any heading, but contains/overlaps a heading, use the first one it contains
        for offset, text, level in heading_offsets:
            if chunk_start <= offset < chunk_end:
                return text

        # 3. Otherwise, fall back to the first heading in the document
        return heading_offsets[0][1]
