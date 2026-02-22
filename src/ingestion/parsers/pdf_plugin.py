"""
ingestion/parsers/pdf_plugin.py
────────────────────────────────
PDF format plugin.
- Text extraction: PyMuPDF (fast, layout-aware)
- Table extraction: pdfplumber + Camelot (lattice tables)
- Chunking: semantic with parent-child

BREAKING CHANGES in PyMuPDF 1.25.x:
  - `import fitz` still works (alias kept for backwards compat) but the
    canonical import is now `import pymupdf as fitz`.
  - `page.get_text("dict")` structure is unchanged.
  - `doc.extract_image(xref)` is unchanged.
  - `block["xref"]` key may be absent on some image blocks — guard with `.get()`.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pandas as pd

from src.core.logging import get_logger
from src.core.models import (
    Chunk, ContentType, DocumentFormat,
    ImageBlock, ParsedDocument, TextBlock,
)
from src.ingestion.chunkers.utils import create_parent_child_chunks, heading_based_chunk
from src.ingestion.parsers.base import BaseFormatPlugin, ChunkConfig

logger = get_logger(__name__)


class PDFPlugin(BaseFormatPlugin):
    """
    PDF parser using PyMuPDF for text and pdfplumber for tables.
    Falls back gracefully if Camelot is unavailable.
    """

    supported_extensions = [".pdf"]
    document_format = DocumentFormat.PDF

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        # PyMuPDF 1.25+: prefer `import pymupdf as fitz` but `import fitz` still works.
        try:
            import pymupdf as fitz          # 1.25+ canonical
        except ImportError:
            import fitz                     # legacy alias

        text_blocks: List[TextBlock] = []
        tables: List[pd.DataFrame] = []
        images: List[ImageBlock] = []
        structure = {"headings": [], "page_count": 0}

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            structure["page_count"] = len(doc)

            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block.get("lines", []):
                            line_text = " ".join(
                                span["text"] for span in line.get("spans", [])
                            ).strip()
                            if not line_text:
                                continue

                            spans = line.get("spans", [])
                            font_size = max((s.get("size", 11) for s in spans), default=11)
                            is_heading = font_size >= 14

                            text_blocks.append(TextBlock(
                                text=line_text,
                                heading=line_text if is_heading else None,
                                page_number=page_num,
                            ))

                            if is_heading:
                                structure["headings"].append(
                                    {"text": line_text, "page": page_num}
                                )

                    elif block["type"] == 1:  # Image block
                        try:
                            # PyMuPDF 1.25: xref may be in block or in "image" sub-dict
                            xref = block.get("xref") or block.get("image", {}).get("xref", 0)
                            if xref:
                                img_data = doc.extract_image(xref)
                                images.append(ImageBlock(
                                    image_bytes=img_data["image"],
                                    format=img_data["ext"],
                                    page_number=page_num,
                                ))
                        except Exception:
                            pass  # Skip unextractable images

        except Exception as e:
            logger.error("PDF text extraction failed", filename=filename, error=str(e))

        # Table extraction with pdfplumber
        tables = self._extract_tables(file_bytes, filename)

        return ParsedDocument(
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            structure=structure,
            source_metadata={"filename": filename, "format": "pdf"},
        )

    def _extract_tables(self, file_bytes: bytes, filename: str) -> List[pd.DataFrame]:
        tables = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = df.dropna(how="all")
                            if not df.empty:
                                tables.append(df)
        except Exception as e:
            logger.warning("Table extraction failed", filename=filename, error=str(e))
        return tables

    def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> List[Chunk]:
        chunks: List[Chunk] = []
        source_meta = doc.source_metadata

        sections = heading_based_chunk(doc.text_blocks, target_tokens=config.parent_chunk_size)

        chunk_index = 0
        for heading, section_text in sections:
            parent, children = create_parent_child_chunks(
                parent_text=section_text,
                heading=heading,
                source_metadata=source_meta,
                child_target_tokens=config.child_chunk_size,
                child_overlap_tokens=config.chunk_overlap,
                chunk_index_offset=chunk_index,
            )
            parent.format = DocumentFormat.PDF
            for child in children:
                child.format = DocumentFormat.PDF
                child.content_type = ContentType.PROSE

            chunks.append(parent)
            chunks.extend(children)
            chunk_index += len(children)

        for i, df in enumerate(doc.tables):
            schema_text = self._table_to_schema_chunk(df, i, source_meta.get("filename", ""))
            if schema_text:
                schema_chunk = Chunk(
                    text=schema_text,
                    source_id=source_meta.get("source_id", ""),
                    content_type=ContentType.TABLE_REF,
                    format=DocumentFormat.PDF,
                )
                chunks.append(schema_chunk)

        logger.info("PDF chunked", filename=source_meta.get("filename"), chunks=len(chunks))
        return chunks

    def _table_to_schema_chunk(self, df: pd.DataFrame, table_idx: int, filename: str) -> str:
        cols = ", ".join(f"{col} ({df[col].dtype})" for col in df.columns)
        sample = df.head(2).to_string(index=False)
        return (
            f"Table {table_idx + 1} from {filename}. "
            f"Columns: {cols}. "
            f"Sample rows:\n{sample}"
        )
