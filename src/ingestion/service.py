"""
ingestion/service.py
─────────────────────
Main ingestion service — orchestrates the complete ingestion pipeline:
intake → parse → chunk → enrich → embed → deduplicate → store
"""
from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path
from typing import List, Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Chunk, DocumentFormat, Namespace, ParsedDocument
from src.ingestion.enricher import MetadataEnricher
from src.ingestion.parsers.base import ChunkConfig, PluginRegistry
from src.storage.cache.redis_store import get_redis_cache
from src.storage.object.minio_store import get_minio_store
from src.storage.sql.postgres_store import get_postgres_store
from src.storage.vector.embedder import get_embedder
from src.storage.vector.qdrant_store import get_qdrant_store

logger = get_logger(__name__)
settings = get_settings()


class IngestionService:
    """
    Orchestrates end-to-end document ingestion.
    Supports: single document, batch ingestion, Kafka consumer.
    """

    def __init__(self):
        self.enricher  = MetadataEnricher()
        self.embedder  = get_embedder()
        self.qdrant    = get_qdrant_store()
        self.pg        = get_postgres_store()
        self.cache     = get_redis_cache()
        self.minio     = get_minio_store()
        self.chunk_config = ChunkConfig(
            child_chunk_size=settings.child_chunk_size,
            parent_chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        source_id: Optional[str] = None,
        namespace_hint: Optional[str] = None,
        doc_date: Optional[str] = None,
        access_roles: Optional[List[str]] = None,
    ) -> dict:
        """
        Ingest a single document through the full pipeline.
        Returns ingestion summary.
        """
        source_id = source_id or str(uuid.uuid4())
        access_roles = access_roles or ["EMPLOYEE"]

        logger.info("Starting ingestion", filename=filename, source_id=source_id)

        try:
            # ── Step 1: Upload raw file to MinIO ─────────────────────────────
            source_url = await self.minio.upload(file_bytes, filename, source_id)

            # ── Step 2: Format detection & parsing ───────────────────────────
            plugin = PluginRegistry.get(filename)
            doc: ParsedDocument = plugin.parse(file_bytes, filename)
            doc.source_metadata.update({
                "source_id": source_id,
                "source_url": source_url,
                "doc_date": doc_date,
            })

            # ── Step 3: Log document intake ───────────────────────────────────
            await self.pg.log_document(
                source_id=source_id,
                filename=filename,
                fmt=doc.format.value,
                namespace=namespace_hint or "GENERAL",
                source_url=source_url,
                doc_date=doc_date,
            )

            # ── Step 4: Store tabular data in PostgreSQL ──────────────────────
            table_count = 0
            for i, df in enumerate(doc.tables):
                sheet = df.attrs.get("sheet_name", f"table_{i}")
                await self.pg.store_dataframe(df, source_id, filename, sheet)
                table_count += 1

            # ── Step 5: Chunking ──────────────────────────────────────────────
            chunks: List[Chunk] = plugin.chunk(doc, self.chunk_config)
            for chunk in chunks:
                chunk.source_id = source_id
                chunk.source_url = source_url
                chunk.doc_date = doc_date
                chunk.access_roles = access_roles
                if namespace_hint:
                    try:
                        chunk.namespace = Namespace(namespace_hint)
                    except ValueError:
                        pass

            logger.info("Chunking complete", filename=filename, chunks=len(chunks))

            # ── Step 6: LLM Metadata Enrichment ──────────────────────────────
            if settings.enrichment_enabled:
                batch_size = settings.enrichment_batch_size
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    await self.enricher.enrich_batch(batch)

            # ── Step 7: Embedding ─────────────────────────────────────────────
            embeddings = self.embedder.embed_chunks(chunks)

            # ── Step 8: Namespace routing + upsert ───────────────────────────
            # Group chunks by namespace (enrichment may have assigned different namespaces)
            namespace_groups: dict[Namespace, List] = {}
            for chunk, embedding in zip(chunks, embeddings):
                ns = chunk.namespace
                if ns not in namespace_groups:
                    namespace_groups[ns] = []
                namespace_groups[ns].append((chunk, embedding))

            total_upserted = 0
            for ns, pairs in namespace_groups.items():
                self.qdrant.ensure_collection(ns)
                ns_chunks = [p[0] for p in pairs]
                ns_embeddings = [p[1] for p in pairs]
                upserted = self.qdrant.upsert_chunks(ns_chunks, ns_embeddings, ns)
                total_upserted += upserted

                # Hot chunk cache for immediate availability
                for chunk in ns_chunks:
                    await self.cache.cache_hot_chunk(chunk.chunk_id, chunk.to_payload())

            # ── Step 9: Update document status ───────────────────────────────
            await self.pg.update_document_status(
                source_id=source_id,
                status="completed",
                chunk_count=total_upserted,
                table_count=table_count,
            )

            result = {
                "source_id": source_id,
                "filename": filename,
                "status": "completed",
                "chunks_ingested": total_upserted,
                "tables_stored": table_count,
                "namespaces": [ns.value for ns in namespace_groups.keys()],
                "source_url": source_url,
            }
            logger.info("Ingestion complete", **result)
            return result

        except Exception as e:
            logger.error("Ingestion failed", filename=filename, error=str(e))
            await self.pg.update_document_status(
                source_id=source_id,
                status="failed",
                error=str(e),
            )
            raise

    async def ingest_batch(self, files: List[dict]) -> List[dict]:
        """
        Ingest multiple files concurrently.
        Each dict: {file_bytes, filename, namespace_hint, doc_date}
        """
        tasks = [
            self.ingest_file(
                file_bytes=f["file_bytes"],
                filename=f["filename"],
                namespace_hint=f.get("namespace_hint"),
                doc_date=f.get("doc_date"),
            )
            for f in files
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def start_kafka_consumer(self) -> None:
        """
        Kafka consumer for event-driven ingestion.
        Listens to rag.ingestion.documents topic.
        """
        from aiokafka import AIOKafkaConsumer
        import json

        consumer = AIOKafkaConsumer(
            settings.kafka_topic_ingestion,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        await consumer.start()
        logger.info("Kafka consumer started", topic=settings.kafka_topic_ingestion)

        try:
            async for msg in consumer:
                event = msg.value
                logger.info("Received ingestion event", filename=event.get("filename"))
                try:
                    # Download from MinIO and ingest
                    file_bytes = await self.minio.download(event["object_key"])
                    await self.ingest_file(
                        file_bytes=file_bytes,
                        filename=event["filename"],
                        source_id=event.get("source_id"),
                        namespace_hint=event.get("namespace"),
                        doc_date=event.get("doc_date"),
                    )
                except Exception as e:
                    logger.error("Kafka event processing failed", event=event, error=str(e))
        finally:
            await consumer.stop()