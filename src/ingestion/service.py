"""
ingestion/service.py
─────────────────────
Main ingestion service — orchestrates the complete ingestion pipeline:
intake → parse → chunk → enrich → embed → deduplicate → store
"""
from __future__ import annotations

import json
import asyncio
import io
import traceback
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

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

            # ── Step 2.1: Resolve namespace ONCE for entire document ───────────
            # Priority: explicit hint → LLM classification → GENERAL fallback
            doc_namespace = await self._resolve_document_namespace(
                doc=doc,
                filename=filename,
                namespace_hint=namespace_hint,
            )
            logger.info(
                "Document namespace resolved",
                filename=filename,
                namespace=doc_namespace.value,
                source=("hint" if namespace_hint else "llm"),
            )

            # ── Step 3: Log document intake ───────────────────────────────────
            await self.pg.log_document(
                source_id=source_id,
                filename=filename,
                fmt=doc.format.value,
                namespace=doc_namespace.value,
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
                chunk.namespace = doc_namespace

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
                "namespaces": [doc_namespace.value],
                "source_url": source_url,
            }
            logger.info("Ingestion complete", **result)
            return result

        except Exception as e:
            logger.error(
                "Ingestion failed",
                filename=filename,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            await self.pg.update_document_status(
                source_id=source_id,
                status="failed",
                error=str(e),
            )
            raise

    async def _resolve_document_namespace(
        self,
        doc: ParsedDocument,
        filename: str,
        namespace_hint: Optional[str] = None,
    ) -> Namespace:
        """
        Resolve the namespace for an entire document — decided ONCE at ingestion.
        
        Priority:
        1. Explicit namespace_hint from caller (highest trust)
        2. LLM classification using document context
        3. GENERAL fallback (if LLM fails)
        """

        # ── Priority 1: Explicit hint from API caller ─────────────────────
        if namespace_hint:
            try:
                ns = Namespace(namespace_hint)
                logger.info(
                    "Namespace resolved from hint",
                    filename=filename,
                    namespace=ns.value,
                )
                return ns
            except ValueError:
                logger.warning(
                    "Invalid namespace hint provided, falling back to LLM",
                    hint=namespace_hint,
                    filename=filename,
                )

        # ── Priority 2: Build document context safely ─────────────────────
        # Different formats produce different ParsedDocument structures:
        # - PDF, Markdown, TXT, PPTX → text_blocks populated, tables maybe
        # - CSV, XLSX               → text_blocks EMPTY, tables populated
        # We must handle both cases explicitly.

        context_parts = [f"Filename: {filename}"]

        # Text-based content (PDF, Markdown, TXT, PPTX)
        if doc.text_blocks:
            sample_text = "\n\n".join(
                block.text for block in doc.text_blocks[:3]
                if block.text.strip()          # skip empty blocks
            )
            if sample_text:
                context_parts.append(f"Content sample:\n{sample_text[:1500]}")

        # Tabular content (CSV, XLSX) — column names + sheet names are strong signal
        if doc.tables:
            for df in doc.tables[:2]:          # first 2 sheets max
                sheet_name = df.attrs.get("sheet_name", "Sheet1")
                col_names = ", ".join(str(c) for c in df.columns[:15])
                sample_rows = df.head(2).to_string(index=False)
                context_parts.append(
                    f"Table sheet: {sheet_name}\n"
                    f"Columns: {col_names}\n"
                    f"Sample rows:\n{sample_rows}"
                )

        # Edge case: completely empty document (no text, no tables)
        if len(context_parts) == 1:
            logger.warning(
                "Document has no extractable content, defaulting to GENERAL",
                filename=filename,
            )
            return Namespace.GENERAL

        doc_context = "\n\n".join(context_parts)

        # ── Priority 3: LLM Classification ───────────────────────────────
        try:
            namespaces_from_db = await self.pg.get_namespace_context()
            namespace_context = json.dumps(namespaces_from_db, indent=2)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a document classifier for an enterprise RAG system.
    Given a document filename and sample content, classify it into exactly ONE namespace.
    Return ONLY a JSON object with one field — no explanation, no markdown.

    Available namespaces (in JSON format):
    {namespace_context}

    JSON schema: {{"namespace": "<namespace>"}}"""),
                ("human", "{doc_context}"),
            ])

            # Use cheaper enrichment model — this is a simple classification task
            llm = ChatOpenAI(
                model=settings.openai_enrichment_model,
                temperature=0.0,
                api_key=settings.openai_api_key,
            )
            chain = prompt | llm | JsonOutputParser()

            
            result = await chain.ainvoke({
                "doc_context": doc_context,
                "namespace_context": namespace_context,
            })

            ns_value = result.get("namespace", "GENERAL")
            ns = Namespace(ns_value)

            logger.info(
                "Namespace resolved via LLM",
                filename=filename,
                namespace=ns.value,
            )
            return ns

        except ValueError as e:
            # LLM returned an unrecognized namespace value
            logger.warning(
                "LLM returned invalid namespace, defaulting to GENERAL",
                filename=filename,
                error=str(e),
            )
            return Namespace.GENERAL

        except Exception as e:
            # LLM call failed entirely
            logger.warning(
                "Namespace LLM classification failed, defaulting to GENERAL",
                filename=filename,
                error=str(e),
            )
            return Namespace.GENERAL

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