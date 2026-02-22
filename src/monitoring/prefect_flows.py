"""
monitoring/prefect_flows.py
────────────────────────────
Prefect flows for scheduled pipeline tasks:
- Weekly index compaction (HNSW defragmentation)
- Nightly batch ingestion from MinIO
- Daily RAGAS evaluation report
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import List

from prefect import flow, task
from prefect.schedules import CronSchedule

from src.core.logging import get_logger, setup_logging
from src.core.models import Namespace
from src.storage.vector.qdrant_store import get_qdrant_store

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Index Compaction
# ─────────────────────────────────────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=60)
def compact_namespace(namespace_value: str) -> dict:
    """Compact and optimize HNSW index for a single namespace."""
    qdrant = get_qdrant_store()
    namespace = Namespace(namespace_value)
    qdrant.optimize_collection(namespace)
    logger.info("Namespace compacted", namespace=namespace_value)
    return {"namespace": namespace_value, "status": "compacted"}


@flow(
    name="weekly-index-compaction",
    description="Compact all Qdrant HNSW indexes to maintain search performance",
)
def weekly_compaction_flow():
    """Run every Sunday at 2 AM."""
    setup_logging()
    results = []
    for namespace in Namespace:
        result = compact_namespace(namespace.value)
        results.append(result)
    logger.info("Weekly compaction complete", namespaces=len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch Ingestion Flow
# ─────────────────────────────────────────────────────────────────────────────

@task(retries=3, retry_delay_seconds=30)
async def ingest_single_file(file_info: dict) -> dict:
    """Ingest a single file task."""
    from src.ingestion.service import IngestionService
    service = IngestionService()
    result = await service.ingest_file(
        file_bytes=file_info["file_bytes"],
        filename=file_info["filename"],
        namespace_hint=file_info.get("namespace"),
        doc_date=file_info.get("doc_date"),
    )
    return result


@flow(
    name="batch-ingestion",
    description="Ingest a batch of documents from MinIO",
)
async def batch_ingestion_flow(file_list: List[dict]):
    """Triggered manually or on schedule for bulk document ingestion."""
    setup_logging()

    # Use Prefect task mapping — submit() returns PrefectFutures that Prefect manages
    futures = [ingest_single_file.submit(f) for f in file_list]

    # Collect results (blocks until all tasks complete)
    results = [f.result(raise_on_failure=False) for f in futures]

    success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
    failed = len(results) - success
    logger.info("Batch ingestion complete", success=success, failed=failed)
    return {"success": success, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# Daily Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────

@task(retries=1)
async def generate_evaluation_report() -> dict:
    """Pull recent RAGAS scores from PostgreSQL and generate a summary."""
    from src.storage.sql.postgres_store import get_postgres_store
    pg = get_postgres_store()

    df = await pg.execute_sql("""
        SELECT
            AVG(faithfulness)       AS avg_faithfulness,
            AVG(answer_relevancy)   AS avg_answer_relevancy,
            AVG(context_precision)  AS avg_context_precision,
            AVG(context_recall)     AS avg_context_recall,
            COUNT(*)                AS total_evaluated
        FROM evaluation_results
        WHERE created_at > NOW() - INTERVAL '24 hours'
    """)

    if df.empty:
        return {"status": "no_data"}

    row = df.iloc[0].to_dict()
    logger.info("Daily evaluation report", **{k: round(float(v), 3) for k, v in row.items() if v})
    return row


@flow(
    name="daily-evaluation-report",
    description="Generate daily RAGAS quality report",
)
async def daily_evaluation_flow():
    """Run every day at 6 AM."""
    setup_logging()
    report = await generate_evaluation_report()
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Deploy Schedules
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Deploy flows with schedules
    weekly_compaction_flow.serve(
        name="weekly-compaction",
        cron="0 2 * * 0",   # Sundays at 2 AM
    )