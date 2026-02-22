"""
core/models.py
──────────────
Central domain models used across the entire RAG pipeline.
All parsers, chunkers, enrichers, and retrievers speak this language.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Namespace(str, Enum):
    """
    Predefined namespaces for domain isolation.
    New namespaces can be added here OR loaded from namespaces.yaml.
    Each namespace maps to a dedicated Qdrant collection.
    """
    HR_EMPLOYEES  = "HR_EMPLOYEES"
    HR_POLICIES   = "HR_POLICIES"
    FINANCE       = "FINANCE"
    TECH_DOCS     = "TECH_DOCS"
    LEGAL         = "LEGAL"
    PRODUCTS      = "PRODUCTS"
    GENERAL       = "GENERAL"


class ContentType(str, Enum):
    PROSE       = "prose"
    TABLE_REF   = "table_ref"    # Schema + sample rows stored as chunk
    SLIDE       = "slide"
    CODE        = "code"
    LIST        = "list"
    EMAIL       = "email"


class QueryType(str, Enum):
    SEMANTIC    = "semantic"     # → vector path
    STRUCTURED  = "structured"   # → SQL path
    HYBRID      = "hybrid"       # → both paths, merge results


class DocumentFormat(str, Enum):
    PDF      = "pdf"
    PPTX     = "pptx"
    XLSX     = "xlsx"
    CSV      = "csv"
    TXT      = "txt"
    MARKDOWN = "markdown"
    DOCX     = "docx"
    HTML     = "html"
    JSON     = "json"
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Raw Parsing Primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """A single text block extracted from a document."""
    text: str
    heading: Optional[str] = None          # Nearest section heading
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    block_index: int = 0
    is_code: bool = False
    is_list: bool = False
    language: str = "en"


@dataclass
class ImageBlock:
    """An image extracted from a document."""
    image_bytes: bytes
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    format: str = "png"


@dataclass
class ParsedDocument:
    """
    Normalized output of every format plugin.
    This is the contract between parsers and the rest of the pipeline.
    Tables go to SQL store. Text blocks go to vector store.
    """
    text_blocks: List[TextBlock]
    tables: List[pd.DataFrame]             # Never embedded — goes to DuckDB/PG
    images: List[ImageBlock]
    structure: Dict[str, Any]              # Headings, outline, slide titles, etc.
    source_metadata: Dict[str, Any]        # filename, format, doc_date, author, etc.

    @property
    def format(self) -> DocumentFormat:
        fmt = self.source_metadata.get("format", "unknown")
        return DocumentFormat(fmt) if fmt in DocumentFormat._value2member_map_ else DocumentFormat.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Chunk — Core Unit of Storage & Retrieval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    The atomic unit stored in the vector database.
    A chunk = text content + rich metadata payload.
    """
    # Content
    text: str
    content_hash: str = field(init=False)

    # Identity
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    source_id: str = ""
    source_url: str = ""
    source_version: int = 1
    is_latest: bool = True

    # Classification
    namespace: Namespace = Namespace.GENERAL
    domain: str = ""
    subdomain: str = ""
    format: DocumentFormat = DocumentFormat.UNKNOWN
    content_type: ContentType = ContentType.PROSE

    # Structure
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1

    # Temporal
    doc_date: Optional[str] = None
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # LLM-enriched metadata (populated by MetadataEnricher)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    hypothetical_questions: List[str] = field(default_factory=list)

    # Quality
    confidence_score: float = 1.0
    language: str = "en"
    token_count: int = 0
    embedding_model: str = ""

    # Access control
    access_roles: List[str] = field(default_factory=lambda: ["EMPLOYEE"])

    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to Qdrant payload dict."""
        return {
            "chunk_id":               self.chunk_id,
            "parent_id":              self.parent_id,
            "source_id":              self.source_id,
            "source_url":             self.source_url,
            "source_version":         self.source_version,
            "is_latest":              self.is_latest,
            "content_hash":           self.content_hash,
            "namespace":              self.namespace.value,
            "domain":                 self.domain,
            "subdomain":              self.subdomain,
            "format":                 self.format.value,
            "content_type":           self.content_type.value,
            "page_number":            self.page_number,
            "slide_number":           self.slide_number,
            "section_heading":        self.section_heading,
            "chunk_index":            self.chunk_index,
            "total_chunks":           self.total_chunks,
            "doc_date":               self.doc_date,
            "ingested_at":            self.ingested_at,
            "created_at":             self.created_at,
            "entities":               self.entities,
            "keywords":               self.keywords,
            "summary":                self.summary,
            "hypothetical_questions": self.hypothetical_questions,
            "confidence_score":       self.confidence_score,
            "language":               self.language,
            "token_count":            self.token_count,
            "embedding_model":        self.embedding_model,
            "access_roles":           self.access_roles,
            # Store text in payload for reranking / context expansion
            "text":                   self.text,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Chunk":
        """Reconstruct a Chunk from a Qdrant payload."""
        c = cls(text=payload["text"])
        for k, v in payload.items():
            if k == "namespace":
                c.namespace = Namespace(v)
            elif k == "format":
                c.format = DocumentFormat(v)
            elif k == "content_type":
                c.content_type = ContentType(v)
            elif hasattr(c, k):
                setattr(c, k, v)
        return c


# ─────────────────────────────────────────────────────────────────────────────
# Query Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    """Output of the Query Classifier — drives the entire retrieval pipeline."""
    original_query: str
    namespaces: List[Namespace]
    query_type: QueryType
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    intent: str = ""
    language: str = "en"
    confidence: float = 1.0


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its relevance score."""
    chunk: Chunk
    score: float
    source: str = "vector"    # "vector" | "sql" | "cache"


@dataclass
class RAGResponse:
    """Final response from the RAG pipeline."""
    answer: str
    source_chunks: List[RetrievedChunk]
    query_plan: QueryPlan
    latency_ms: float
    cached: bool = False
    evaluation_scores: Optional[Dict[str, float]] = None