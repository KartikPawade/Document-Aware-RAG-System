"""
api/app.py
───────────
FastAPI application exposing:
- POST /ingest          — upload and ingest a document
- POST /query           — query the RAG system
- GET  /health          — health check
- GET  /namespaces      — list available namespaces
- GET  /documents       — list ingested documents
- GET  /metrics         — Prometheus metrics
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.core.models import Namespace
from src.ingestion.service import IngestionService
from src.retrieval.pipeline import RetrievalPipeline
from src.storage.sql.postgres_store import get_postgres_store
from src.storage.vector.qdrant_store import get_qdrant_store

logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────

INGESTION_COUNTER = Counter(
    "rag_ingestion_total",
    "Total documents ingested",
    ["format", "namespace", "status"],
)
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total queries processed",
    ["query_type", "cached"],
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query latency in seconds",
    ["query_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "Reranker scores for retrieved chunks",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting RAG API", environment=settings.environment)

    # Initialize storage
    pg = get_postgres_store()
    await pg.initialize()

    qdrant = get_qdrant_store()
    qdrant.ensure_all_collections()

    logger.info("RAG API ready")
    yield
    logger.info("RAG API shutting down")


app = FastAPI(
    title="Advanced RAG API",
    description="Production-grade multi-format RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Services (lazy-initialized)
_ingestion_service: Optional[IngestionService] = None
_retrieval_pipeline: Optional[RetrievalPipeline] = None

def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service

def get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    user_roles: List[str] = ["EMPLOYEE"]
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    query_type: str
    namespaces: List[str]
    sources: List[dict]
    latency_ms: float
    cached: bool

class IngestResponse(BaseModel):
    source_id: str
    filename: str
    status: str
    chunks_ingested: int
    tables_stored: int
    namespaces: List[str]
    source_url: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "supported_formats": [
            ".pdf", ".pptx", ".xlsx", ".csv", ".txt", ".md"
        ],
    }


@app.get("/namespaces")
async def list_namespaces():
    """List all available namespaces with descriptions."""
    return {
        "namespaces": [
            {"id": ns.value, "description": _ns_descriptions().get(ns.value, "")}
            for ns in Namespace
        ]
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    namespace: Optional[str] = Form(None),
    doc_date: Optional[str] = Form(None),
    access_roles: Optional[str] = Form("EMPLOYEE"),
):
    """
    Upload and ingest a document.
    Supported formats: PDF, PPTX, XLSX, CSV, TXT, Markdown.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400,
            detail="File must have a filename (e.g. document.pdf).",
        )
    file_bytes = await file.read()
    roles = [r.strip() for r in (access_roles or "EMPLOYEE").split(",")]

    try:
        service = get_ingestion_service()
        result = await service.ingest_file(
            file_bytes=file_bytes,
            filename=file.filename,
            namespace_hint=namespace,
            doc_date=doc_date,
            access_roles=roles,
        )

        INGESTION_COUNTER.labels(
            format=result.get("filename", "").rsplit(".", 1)[-1],
            namespace=",".join(result.get("namespaces", [])),
            status="success",
        ).inc()

        return IngestResponse(**result)

    except ValueError as e:
        INGESTION_COUNTER.labels(format="unknown", namespace="unknown", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        INGESTION_COUNTER.labels(format="unknown", namespace="unknown", status="error").inc()
        logger.error("Ingestion endpoint error", error=str(e))
        raise HTTPException(status_code=500, detail="Ingestion failed")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with natural language.
    Automatically routes to semantic or SQL retrieval path.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        pipeline = get_retrieval_pipeline()
        start = time.time()

        response = await pipeline.query(
            user_query=request.query,
            user_roles=request.user_roles,
            top_k=request.top_k,
        )

        latency = (time.time() - start) * 1000

        # Record metrics
        QUERY_COUNTER.labels(
            query_type=response.query_plan.query_type.value,
            cached=str(response.cached),
        ).inc()
        QUERY_LATENCY.labels(
            query_type=response.query_plan.query_type.value
        ).observe(latency / 1000)

        for rc in response.source_chunks:
            RETRIEVAL_SCORE.observe(rc.score)

        return QueryResponse(
            answer=response.answer,
            query_type=response.query_plan.query_type.value,
            namespaces=[ns.value for ns in response.query_plan.namespaces],
            sources=[
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "source_url": rc.chunk.source_url,
                    "section": rc.chunk.section_heading,
                    "score": round(rc.score, 4),
                    "namespace": rc.chunk.namespace.value,
                }
                for rc in response.source_chunks
            ],
            latency_ms=round(latency, 2),
            cached=response.cached,
        )

    except Exception as e:
        logger.error("Query endpoint error", query=request.query[:60], error=str(e))
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/documents")
async def list_documents(
    namespace: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all ingested documents with their status."""
    pg = get_postgres_store()
    async with pg._session_factory() as session:
        from sqlalchemy import text
        where = "WHERE namespace = :ns" if namespace else ""
        result = await session.execute(
            text(f"""
                SELECT source_id, filename, format, namespace, status,
                       chunk_count, table_count, ingested_at
                FROM ingested_documents
                {where}
                ORDER BY ingested_at DESC
                LIMIT :limit OFFSET :offset
            """),
            {"ns": namespace, "limit": limit, "offset": offset},
        )
        rows = result.fetchall()
        return {
            "documents": [
                {
                    "source_id": str(r[0]),
                    "filename": r[1],
                    "format": r[2],
                    "namespace": r[3],
                    "status": r[4],
                    "chunk_count": r[5],
                    "table_count": r[6],
                    "ingested_at": str(r[7]),
                }
                for r in rows
            ]
        }


def _ns_descriptions() -> dict:
    return {
        "HR_EMPLOYEES": "Employee profiles, org charts, roles",
        "HR_POLICIES": "Company policies, handbooks, PTO rules",
        "FINANCE": "Budgets, forecasts, invoices",
        "TECH_DOCS": "API docs, architecture, runbooks",
        "LEGAL": "Contracts, NDAs, compliance",
        "PRODUCTS": "Product specs, roadmaps, release notes",
        "GENERAL": "General documents",
    }