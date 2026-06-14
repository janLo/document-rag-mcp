from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
import fitz  # PyMuPDF
import yaml

fitz.TOOLS.mupdf_display_errors(False)


@dataclass
class ExtractedPage:
    text: str
    page_number: int | None = None
    headings: list[tuple[str, int]] = field(default_factory=list)  # list of (heading_text, heading_level)
    image_bytes: bytes | None = None
    metadata: dict[str, any] = field(default_factory=dict)


class DocumentExtractor:
    def __init__(self, vision_enabled: bool = False):
        self.vision_enabled = vision_enabled

    def extract(self, file_path: Path | str) -> list[ExtractedPage]:
        """Extracts content from a file depending on its extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".txt":
            return self._extract_txt(path)
        elif ext == ".md":
            return self._extract_md(path)
        elif ext == ".pdf":
            return self._extract_pdf(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _extract_txt(self, path: Path) -> list[ExtractedPage]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        headings = []
        lines = content.splitlines()
        if lines:
            first_line = lines[0].strip()
            # If first line looks like a title, mark it as heading level 1
            if first_line and len(first_line) < 80 and not first_line.endswith((".", ",", ";")):
                headings.append((first_line, 1))

        return [
            ExtractedPage(
                text=content,
                page_number=1,
                headings=headings,
                metadata={"title": path.stem},
            )
        ]

    def _extract_md(self, path: Path) -> list[ExtractedPage]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        frontmatter = {}
        body = content

        # Parse frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2]
                except Exception:
                    pass

        headings = []
        for line in body.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("#"):
                # Count the heading level
                level = 0
                while level < len(line_strip) and line_strip[level] == "#":
                    level += 1
                if level > 0 and level < len(line_strip) and line_strip[level] == " ":
                    heading_text = line_strip[level:].strip()
                    if heading_text:
                        headings.append((heading_text, level))

        # Default title is the frontmatter title, or file stem
        title = frontmatter.get("title", path.stem)
        meta = {"title": title, "frontmatter": frontmatter}

        return [
            ExtractedPage(
                text=body,
                page_number=1,
                headings=headings,
                metadata=meta,
            )
        ]

    def _extract_pdf(self, path: Path) -> list[ExtractedPage]:
        doc = fitz.open(path)
        pages = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            
            # Use blocks to extract text as logical paragraphs, cleaning up intra-paragraph line breaks
            try:
                blocks = page.get_text("blocks")
                text_blocks = []
                for b in blocks:
                    if len(b) >= 7 and b[6] == 0:  # text block
                        # Fix hyphenated words across lines, then replace other newlines with space
                        block_text = b[4].strip()
                        block_text = re.sub(r"-\n\s*", "", block_text)
                        block_text = re.sub(r"\s*\n\s*", " ", block_text)
                        if block_text:
                            text_blocks.append(block_text)
            except Exception:
                # Fallback to visual text extraction if the structure tree is malformed
                raw_text = page.get_text("text").strip()
                text_blocks = [raw_text] if raw_text else []
            
            text = "\n\n".join(text_blocks)

            # Scanned or Diagram check
            # If the page has very little text overall OR the text consists of many small fragmented blocks (like diagram labels)
            non_space_chars = len("".join(text.split()))
            
            avg_block_len = 0
            if len(text_blocks) > 0:
                avg_block_len = sum(len(b) for b in text_blocks) / len(text_blocks)
                
            is_empty_or_diagram = (non_space_chars < 40) or (0 < avg_block_len < 25)

            image_bytes = None
            if is_empty_or_diagram and self.vision_enabled:
                # Render to high-quality PNG (144 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

            # Typography-aware heading detection using get_text("dict")
            headings = []
            try:
                blocks = page.get_text("dict").get("blocks", [])
                sizes = []
                for b in blocks:
                    if b.get("type") == 0:  # text block
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                txt = span.get("text", "").strip()
                                if txt:
                                    sizes.append(span.get("size", 10.0))

                # Compute body text font size (most common size)
                body_size = 10.0
                if sizes:
                    body_size = Counter(sizes).most_common(1)[0][0]

                # Detect headings
                for b in blocks:
                    if b.get("type") == 0:
                        for line in b.get("lines", []):
                            spans = line.get("spans", [])
                            if not spans:
                                continue
                            line_text = "".join(s.get("text", "") for s in spans).strip()
                            if not line_text:
                                continue

                            first_span = spans[0]
                            size = first_span.get("size", 10.0)
                            font = first_span.get("font", "").lower()

                            is_bold = "bold" in font or "black" in font or "heavy" in font
                            is_large = size > body_size * 1.2

                            # Short line, doesn't end with typical sentence punctuation
                            if (
                                (is_large or is_bold)
                                and len(line_text) < 120
                                and not line_text.endswith((".", ":", ";", ","))
                            ):
                                if size > body_size * 1.5:
                                    level = 1
                                elif size > body_size * 1.3:
                                    level = 2
                                else:
                                    level = 3
                                headings.append((line_text, level))
            except Exception:
                # Fallback: if dict parsing fails, don't break extraction
                pass

            pages.append(
                ExtractedPage(
                    text=text,
                    page_number=page_num,
                    headings=headings,
                    image_bytes=image_bytes,
                    metadata={"title": path.stem},
                )
            )

        doc.close()
        return pages
