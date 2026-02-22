"""
core/config.py
──────────────
Pydantic settings loaded from environment variables / .env file.
All tunable parameters for the RAG pipeline live here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    environment: str = Field(default="development", description="development | staging | production")

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Model for answer generation and query classification")
    openai_enrichment_model: str = Field(default="gpt-4o-mini", description="Cheaper model for metadata enrichment")

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_prefer_grpc: bool = Field(default=False, description="Use gRPC for Qdrant (faster for large payloads)")
    dense_vector_size: int = Field(default=1024, description="BGE-M3 dense vector dimensions")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model for dense + sparse embeddings",
    )
    embedding_device: str = Field(default="cpu", description="'cpu' | 'cuda' | 'mps'")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding calls")

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder reranker model",
    )
    reranker_top_n: int = Field(default=5, description="Number of chunks to return after reranking")
    reranker_batch_size: int = Field(default=16, description="Batch size for reranker scoring")

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_dsn: str = Field(
        default="postgresql://raguser:ragpassword@localhost:5432/ragdb",
        description="PostgreSQL connection string (sync DSN — asyncpg variant auto-derived)",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    query_cache_ttl: int = Field(default=1800, description="Query response cache TTL in seconds (30 min)")
    hot_chunk_ttl: int = Field(default=300, description="Hot chunk cache TTL in seconds (5 min)")

    # ── MinIO / S3 ────────────────────────────────────────────────────────────
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint (host:port)")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin123", description="MinIO secret key")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO")
    minio_bucket_documents: str = Field(default="rag-documents", description="Bucket for raw source documents")

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    kafka_topic_ingestion: str = Field(default="rag.ingestion.documents", description="Kafka topic for ingestion events")
    kafka_consumer_group: str = Field(default="rag-ingestion-group", description="Kafka consumer group ID")

    # ── Ingestion Pipeline ────────────────────────────────────────────────────
    child_chunk_size: int = Field(default=400, description="Target token size for child chunks")
    parent_chunk_size: int = Field(default=1500, description="Target token size for parent sections")
    chunk_overlap: int = Field(default=50, description="Overlap in tokens between adjacent chunks")

    enrichment_enabled: bool = Field(default=True, description="Enable LLM metadata enrichment")
    enrichment_batch_size: int = Field(default=10, description="Concurrent enrichment calls per batch")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, description="Candidates retrieved from hybrid search")
    min_similarity_score: float = Field(default=0.72, description="Minimum score threshold — below this returns 'no info'")
    max_context_tokens: int = Field(default=4000, description="Hard cap on total context tokens sent to LLM")

    # ── Evaluation (RAGAS) ───────────────────────────────────────────────────
    ragas_enabled: bool = Field(default=True, description="Enable online RAGAS sampling")
    ragas_sample_rate: float = Field(default=0.01, description="Fraction of queries evaluated online (1%)")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings instance (cached after first call)."""
    return Settings()