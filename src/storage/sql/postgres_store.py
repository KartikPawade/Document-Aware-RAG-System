"""
storage/sql/postgres_store.py
──────────────────────────────
PostgreSQL store for:
1. Tabular data from XLSX/CSV (queried via NL→SQL)
2. Chunk metadata index (audit trail, relationships)
3. Document ingestion log

CHANGES for asyncpg 0.30.x + SQLAlchemy 2.0.36 + PostgreSQL 17:
  - asyncpg 0.30 dropped support for Python 3.8/3.9; 3.12 is fully supported.
  - `create_async_engine`: `pool_pre_ping=True` is recommended for PG 17
    to detect stale connections after PG server restarts.
  - PostgreSQL 17: no SQL syntax changes that affect this schema.
  - `async_sessionmaker` is stable in SQLAlchemy 2.0.36 (no breaking changes).
  - `pd.DataFrame.to_sql()` with a sync SQLAlchemy engine: SQLAlchemy 2.x
    requires `con` to be a `Connection` or `Engine` — using `Engine` is fine.
  - The `method="multi"` kwarg in `to_sql()` is still valid in pandas 2.2.x.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PostgresStore:
    """Async PostgreSQL store for tabular data and document metadata."""

    def __init__(self):
        # Convert sync DSN to async DSN
        dsn = settings.postgres_dsn.replace("postgresql://", "postgresql+asyncpg://")
        self._engine = create_async_engine(
            dsn,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Detect stale connections (important for PG 17 + long-lived pools)
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create all required tables on startup."""
        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    id              SERIAL PRIMARY KEY,
                    source_id       UUID NOT NULL UNIQUE,
                    filename        TEXT NOT NULL,
                    format          TEXT NOT NULL,
                    namespace       TEXT NOT NULL,
                    source_url      TEXT,
                    doc_date        DATE,
                    version         INTEGER DEFAULT 1,
                    chunk_count     INTEGER DEFAULT 0,
                    table_count     INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'processing',
                    error_message   TEXT,
                    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """))

            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tabular_datasets (
                    id              SERIAL PRIMARY KEY,
                    source_id       UUID NOT NULL,
                    table_name      TEXT NOT NULL UNIQUE,
                    original_file   TEXT NOT NULL,
                    sheet_name      TEXT,
                    column_schema   JSONB,
                    row_count       INTEGER,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    FOREIGN KEY (source_id) REFERENCES ingested_documents(source_id)
                        ON DELETE CASCADE
                )
            """))

            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id              SERIAL PRIMARY KEY,
                    query_text      TEXT NOT NULL,
                    query_type      TEXT,
                    namespaces      TEXT[],
                    latency_ms      FLOAT,
                    chunk_count     INTEGER,
                    cached          BOOLEAN DEFAULT FALSE,
                    user_id         TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """))

            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id              SERIAL PRIMARY KEY,
                    query_text      TEXT NOT NULL,
                    answer          TEXT,
                    faithfulness    FLOAT,
                    answer_relevancy FLOAT,
                    context_precision FLOAT,
                    context_recall  FLOAT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """))

            # ── Namespace Descriptions ────────────────────────────────────
            # Single source of truth for namespace descriptions.
            # Used by all LLM prompts: classifier, ingestion service.
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS namespace_descriptions (
                    namespace    TEXT PRIMARY KEY,
                    description  TEXT NOT NULL,
                    is_active    BOOLEAN DEFAULT TRUE,
                    created_at   TIMESTAMPTZ DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """))

            # Seed default namespace descriptions
            # ON CONFLICT DO NOTHING — never overwrites manual updates
            await conn.execute(text("""
                INSERT INTO namespace_descriptions (namespace, description) VALUES
                (
                    'HR_EMPLOYEES',
                    'Employee profiles, org charts, roles, seniority levels, performance reviews, salary bands, certifications, tech stacks per employee, team assignments, engineer data, staff headcount'
                ),
                (
                    'HR_POLICIES',
                    'Company policies, employee handbooks, PTO rules, code of conduct, leave policies, HR procedures, workplace guidelines, benefits documentation'
                ),
                (
                    'FINANCE',
                    'Budgets, financial forecasts, invoices, expense reports, financial statements, revenue data, cost analysis, profit and loss, accounting records'
                ),
                (
                    'TECH_DOCS',
                    'API documentation, system architecture, runbooks, deployment guides, infrastructure specs, security documentation, technical specifications, code documentation, developer guides'
                ),
                (
                    'LEGAL',
                    'Contracts, NDAs, compliance documents, regulatory filings, legal agreements, privacy policies, terms of service, intellectual property, legal correspondence'
                ),
                (
                    'PRODUCTS',
                    'Product specifications, pricing, feature lists, roadmaps, release notes, SLA policies, support tiers, product FAQs, deployment options, enterprise offerings, product comparisons'
                ),
                (
                    'GENERAL',
                    'General documents that do not clearly fit other categories, miscellaneous content, cross-domain documents'
                )
                ON CONFLICT (namespace) DO NOTHING
            """))

            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_docs_source_id ON ingested_documents(source_id)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_docs_namespace ON ingested_documents(namespace)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_tabular_source ON tabular_datasets(source_id)"
            ))
        logger.info("PostgreSQL tables initialized")



    async def get_namespace_context(self) -> list[dict]:
        """
        Fetch all active namespace descriptions from DB.
        Returns a list of dicts — callers serialize as needed for prompts.

        Output:
            [
                {"namespace": "HR_EMPLOYEES", "description": "Employee profiles..."},
                {"namespace": "HR_POLICIES",  "description": "Company policies..."},
                ...
            ]
        """
        async with self._session_factory() as session:
            result = await session.execute(text("""
                SELECT namespace, description
                FROM namespace_descriptions
                WHERE is_active = TRUE
                ORDER BY namespace
            """))
            rows = result.fetchall()
            return [
                {
                    "namespace":   row[0],
                    "description": row[1],
                }
                for row in rows
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # Document Registry
    # ─────────────────────────────────────────────────────────────────────────

    async def log_document(
        self,
        source_id: str,
        filename: str,
        fmt: str,
        namespace: str,
        source_url: str = "",
        doc_date: Optional[str] = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(text("""
                INSERT INTO ingested_documents
                    (source_id, filename, format, namespace, source_url, doc_date, status)
                VALUES
                    (:source_id, :filename, :format, :namespace, :source_url, :doc_date, 'processing')
                ON CONFLICT (source_id) DO UPDATE SET
                    version = ingested_documents.version + 1,
                    updated_at = NOW()
            """), {
                "source_id": source_id, "filename": filename, "format": fmt,
                "namespace": namespace, "source_url": source_url, "doc_date": doc_date,
            })
            await session.commit()

    async def update_document_status(
        self,
        source_id: str,
        status: str,
        chunk_count: int = 0,
        table_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(text("""
                UPDATE ingested_documents SET
                    status = :status,
                    chunk_count = :chunk_count,
                    table_count = :table_count,
                    error_message = :error,
                    updated_at = NOW()
                WHERE source_id = CAST(:source_id AS uuid)
            """), {
                "source_id": source_id, "status": status,
                "chunk_count": chunk_count, "table_count": table_count, "error": error,
            })
            await session.commit()

    # ─────────────────────────────────────────────────────────────────────────
    # Tabular Data Storage
    # ─────────────────────────────────────────────────────────────────────────

    async def store_dataframe(
        self,
        df: pd.DataFrame,
        source_id: str,
        original_file: str,
        sheet_name: str = "default",
    ) -> str:
        """
        Store a DataFrame as a PostgreSQL table.
        Table name is derived from source_id + sheet_name.
        Returns the table name for future SQL queries.
        """
        import hashlib

        # Sanitize columns — pdfplumber can produce None or duplicate headers
        sanitized = []
        seen = {}
        for i, col in enumerate(df.columns):
            col = str(col).strip() if col is not None else f"col_{i}"
            col = col if col else f"col_{i}"
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            sanitized.append(col)
        df.columns = sanitized

        table_name = "tbl_" + hashlib.md5(
            f"{source_id}_{sheet_name}".encode()
        ).hexdigest()[:12]

        import asyncio

        def _sync_to_sql():
            import sqlalchemy
            # SQLAlchemy 2.x: create_engine with future=True is the default;
            # pandas to_sql() accepts Engine directly.
            sync_engine = sqlalchemy.create_engine(settings.postgres_dsn)
            df.to_sql(
                table_name,
                sync_engine,
                if_exists="replace",
                index=False,
                method="multi",
            )
            sync_engine.dispose()

        await asyncio.to_thread(_sync_to_sql)

        column_schema = {col: str(df[col].dtype) for col in df.columns}
        async with self._session_factory() as session:
            await session.execute(text("""
                INSERT INTO tabular_datasets
                    (source_id, table_name, original_file, sheet_name, column_schema, row_count)
                VALUES
                    (:source_id, :table_name, :original_file, :sheet_name, :column_schema, :row_count)
                ON CONFLICT (table_name) DO UPDATE SET
                    row_count = :row_count,
                    column_schema = :column_schema
            """), {
                "source_id": source_id,
                "table_name": table_name,
                "original_file": original_file,
                "sheet_name": sheet_name,
                "column_schema": json.dumps(column_schema),
                "row_count": len(df),
            })
            await session.commit()

        logger.info("DataFrame stored", table=table_name, rows=len(df))
        return table_name

    async def get_table_schema(self, source_id: str) -> List[Dict[str, Any]]:
        """Get schemas of all tables (used for NL→SQL context).
        If source_id is empty, returns all tables."""
        async with self._session_factory() as session:
            if source_id:
                result = await session.execute(text("""
                    SELECT table_name, sheet_name, column_schema, row_count
                    FROM tabular_datasets
                    WHERE source_id = CAST(:source_id AS uuid)
                """), {"source_id": source_id})
            else:
                result = await session.execute(text("""
                    SELECT table_name, sheet_name, column_schema, row_count
                    FROM tabular_datasets
                    ORDER BY created_at DESC
                    LIMIT 50
                """))
            rows = result.fetchall()
            return [
                {
                    "table_name": r[0],
                    "sheet_name": r[1],
                    "column_schema": (
                        r[2] if isinstance(r[2], dict)          # asyncpg returns dict natively
                        else json.loads(r[2]) if r[2]           # fallback: parse string
                        else {}
                    ),
                    "row_count": r[3],
                }
                for r in rows
            ]

    async def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        async with self._session_factory() as session:
            result = await session.execute(text(sql))
            rows = result.fetchall()
            cols = list(result.keys())
            return pd.DataFrame(rows, columns=cols)

    # ─────────────────────────────────────────────────────────────────────────
    # Query Logging
    # ─────────────────────────────────────────────────────────────────────────

    async def log_query(
        self,
        query_text: str,
        query_type: str,
        namespaces: List[str],
        latency_ms: float,
        chunk_count: int,
        cached: bool,
        user_id: Optional[str] = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(text("""
                INSERT INTO query_logs
                    (query_text, query_type, namespaces, latency_ms, chunk_count, cached, user_id)
                VALUES
                    (:query_text, :query_type, :namespaces, :latency_ms, :chunk_count, :cached, :user_id)
            """), {
                "query_text": query_text, "query_type": query_type,
                "namespaces": namespaces, "latency_ms": latency_ms,
                "chunk_count": chunk_count, "cached": cached, "user_id": user_id,
            })
            await session.commit()

    async def log_evaluation(self, results: Dict[str, Any]) -> None:
        async with self._session_factory() as session:
            await session.execute(text("""
                INSERT INTO evaluation_results
                    (query_text, answer, faithfulness, answer_relevancy, context_precision, context_recall)
                VALUES
                    (:query_text, :answer, :faithfulness, :answer_relevancy, :context_precision, :context_recall)
            """), results)
            await session.commit()


# Singleton
_pg_store: Optional[PostgresStore] = None

def get_postgres_store() -> PostgresStore:
    global _pg_store
    if _pg_store is None:
        _pg_store = PostgresStore()
    return _pg_store
