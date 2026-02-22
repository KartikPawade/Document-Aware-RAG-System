"""
storage/vector/qdrant_store.py
───────────────────────────────
Qdrant vector store with namespace isolation.
Each namespace = one Qdrant collection.
Supports: hybrid search (dense + sparse RRF), metadata pre-filtering,
deduplication, versioning, collection creation/management.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance, FieldCondition, Filter, MatchAny, MatchValue,
    PointStruct, Prefetch, FusionQuery, Fusion, Range,
    SparseVector, VectorParams, SparseVectorParams, SparseIndexParams,
    PayloadSchemaType,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Chunk, Namespace, QueryPlan, RetrievedChunk
from src.storage.vector.embedder import EmbeddingResult, get_embedder

logger = get_logger(__name__)
settings = get_settings()


class QdrantStore:
    """
    Manages all interactions with the Qdrant vector database.
    Handles: collection setup, upsert, hybrid search, deduplication.
    """

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
        return self._client

    def _get_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        return self._async_client

    # ─────────────────────────────────────────────────────────────────────────
    # Collection Management
    # ─────────────────────────────────────────────────────────────────────────

    def ensure_collection(self, namespace: Namespace) -> None:
        """
        Create collection for namespace if it doesn't exist.
        Each collection has: dense vector, sparse vector, indexed payload fields.
        """
        client = self._get_client()
        collection_name = namespace.value

        try:
            client.get_collection(collection_name)
            logger.debug("Collection exists", collection=collection_name)
            return
        except Exception:
            pass  # Collection doesn't exist, create it

        logger.info("Creating Qdrant collection", collection=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

        # Create payload indexes for fast pre-filtering
        indexed_fields = {
            "namespace":         PayloadSchemaType.KEYWORD,
            "is_latest":         PayloadSchemaType.BOOL,
            "language":          PayloadSchemaType.KEYWORD,
            "format":            PayloadSchemaType.KEYWORD,
            "content_type":      PayloadSchemaType.KEYWORD,
            "doc_date":          PayloadSchemaType.KEYWORD,
            "entities":          PayloadSchemaType.KEYWORD,
            "access_roles":      PayloadSchemaType.KEYWORD,
            "source_id":         PayloadSchemaType.KEYWORD,
            "content_hash":      PayloadSchemaType.KEYWORD,
            "confidence_score":  PayloadSchemaType.FLOAT,
            "chunk_index":       PayloadSchemaType.INTEGER,
        }

        for field, schema_type in indexed_fields.items():
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("Collection created with indexes", collection=collection_name)

    def ensure_all_collections(self) -> None:
        """Create all namespace collections on startup."""
        for namespace in Namespace:
            self.ensure_collection(namespace)

    # ─────────────────────────────────────────────────────────────────────────
    # Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def find_by_hash(self, content_hash: str, namespace: Namespace) -> Optional[Dict]:
        """Check if a chunk with this content hash already exists."""
        client = self._get_client()
        results, _ = client.scroll(
            collection_name=namespace.value,
            scroll_filter=Filter(must=[
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash))
            ]),
            limit=1,
            with_payload=True,
        )
        return results[0].payload if results else None

    def soft_delete_by_source(self, source_id: str, namespace: Namespace) -> int:
        """Mark all chunks from a source as not latest (for versioning)."""
        client = self._get_client()
        client.set_payload(
            collection_name=namespace.value,
            payload={"is_latest": False},
            points_selector=Filter(must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id))
            ]),
        )
        logger.info("Soft-deleted old version chunks", source_id=source_id, namespace=namespace)
        return 1  # Qdrant doesn't return count for set_payload

    # ─────────────────────────────────────────────────────────────────────────
    # Upsert
    # ─────────────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
        namespace: Namespace,
    ) -> int:
        """
        Upsert chunks with their embeddings into the correct namespace collection.
        Handles deduplication before upsert.
        """
        client = self._get_client()
        points = []
        skipped = 0

        for chunk, embedding in zip(chunks, embeddings):
            # Deduplication check
            existing = self.find_by_hash(chunk.content_hash, namespace)
            if existing:
                if existing.get("source_version", 0) >= chunk.source_version:
                    skipped += 1
                    continue
                # New version — soft-delete old
                self.soft_delete_by_source(chunk.source_id, namespace)

            # Update embedding model field
            chunk.embedding_model = settings.embedding_model_name
            chunk.token_count = embedding.token_count

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": embedding.dense,
                    "sparse": SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ) if embedding.sparse else SparseVector(indices=[], values=[]),
                },
                payload=chunk.to_payload(),
            )
            points.append(point)

        if points:
            client.upsert(collection_name=namespace.value, points=points)
            logger.info(
                "Chunks upserted",
                collection=namespace.value,
                count=len(points),
                skipped=skipped,
            )

        return len(points)

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid Search
    # ─────────────────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        namespace: Namespace,
        prefilter: Optional[Filter],
        top_k: int = 20,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search: dense + sparse with RRF fusion.
        Pre-filter is applied before ANN search (not post-hoc).
        """
        client = self._get_client()

        try:
            results = client.query_points(
                collection_name=namespace.value,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k * 2,
                        filter=prefilter,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using="sparse",
                        limit=top_k * 2,
                        filter=prefilter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points

        except Exception as e:
            logger.warning(
                "Hybrid search failed, falling back to dense-only",
                namespace=namespace.value,
                error=str(e),
            )
            # Use query_points (client 1.17+); search() was removed / differs by version
            results = client.query_points(
                collection_name=namespace.value,
                query=dense_vector,
                using="dense",
                query_filter=prefilter,
                limit=top_k,
                with_payload=True,
            ).points

        retrieved = []
        for point in results:
            chunk = Chunk.from_payload(point.payload)
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=point.score,
                source="vector",
            ))

        return retrieved

    def get_chunk_by_id(self, chunk_id: str, namespace: Namespace) -> Optional[Chunk]:
        """Fetch a single chunk by ID (used for parent context expansion)."""
        client = self._get_client()
        results, _ = client.scroll(
            collection_name=namespace.value,
            scroll_filter=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
            ]),
            limit=1,
            with_payload=True,
        )
        if results:
            return Chunk.from_payload(results[0].payload)
        return None

    def optimize_collection(self, namespace: Namespace) -> None:
        """Trigger HNSW index optimization (run weekly via Prefect)."""
        client = self._get_client()
        client.update_collection(
            collection_name=namespace.value,
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=0),
        )
        logger.info("Collection optimization triggered", collection=namespace.value)


# Singleton
_store: Optional[QdrantStore] = None

def get_qdrant_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store