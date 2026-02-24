"""
retrieval/nl_to_sql.py
───────────────────────
NL→SQL engine for the structured query path.
Uses LCEL: prompt | llm | StrOutputParser()
Executes against PostgreSQL with safety guardrails (SELECT only).

When relevant_tables is provided by the router, only those tables are
included in the schema context — keeps the prompt focused and accurate.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger
from src.storage.sql.postgres_store import get_postgres_store

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# NL→SQL Prompt — LCEL
# ─────────────────────────────────────────────────────────────────────────────

NL_TO_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. Convert the natural language query to a PostgreSQL SELECT statement.

Rules:
- Only generate SELECT statements — never INSERT, UPDATE, DELETE, DROP, or any DDL
- Use exact column names from the schema provided
- Use ILIKE for case-insensitive text matching
- If the query cannot be answered from the schema, return: SELECT 'NO_DATA' AS result
- Return ONLY the SQL statement — no explanation, no markdown, no backticks

Available tables and schemas:
{schema_context}"""),
    ("human", "Natural language query: {query}"),
])


class NLToSQLEngine:
    """
    Converts natural language queries to SQL and executes them.
    Activated for QueryType.STRUCTURED and the SQL leg of QueryType.HYBRID.

    When the router provides relevant_tables, we build a focused schema
    context from the registry rather than including every table — this
    produces more accurate SQL and a shorter prompt.
    """

    def __init__(self):
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        # LCEL chain: prompt → LLM → string output
        self._chain = NL_TO_SQL_PROMPT | llm | StrOutputParser()
        self._pg = get_postgres_store()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
    async def execute(
        self,
        query: str,
        schema_context: str,
        relevant_tables: Optional[List[str]] = None,
    ) -> str:
        """
        Generate SQL from natural language, execute it, return formatted result.

        Args:
            query:           The natural language query.
            schema_context:  Full schema context string (used as fallback).
            relevant_tables: Table names from the router's RouteDecision.
                             If provided, schema_context is rebuilt from the
                             registry scoped to only these tables — keeping
                             the prompt focused and the SQL accurate.

        Returns:
            A string context block suitable for passing to the final LLM.
        """
        # ── Build focused schema context from registry if tables are known ──
        if relevant_tables:
            registry = await self._pg.get_schema_registry()
            focused_lines = []
            for t in relevant_tables:
                if t in registry:
                    info = registry[t]
                    cols = ", ".join(info["columns"])
                    focused_lines.append(
                        f"Table `{t}`: columns=[{cols}] | source={info['source']} "
                        f"| rows={info['row_count']} | {info['description']}"
                    )
            if focused_lines:
                schema_context = "\n".join(focused_lines)
                logger.info(
                    "Using focused schema context",
                    relevant_tables=relevant_tables,
                    table_count=len(focused_lines),
                )

        try:
            # ── Generate SQL via LCEL ──────────────────────────────────────
            sql = await self._chain.ainvoke({
                "query":          query,
                "schema_context": schema_context,
            })

            # Safety: strip markdown if LLM wrapped in backticks
            sql = self._sanitize_sql(sql)

            # ── LOG 1: Generated SQL ───────────────────────────────────────
            logger.info(
                "SQL generated",
                natural_language_query=query,
                generated_sql=sql,
            )

            if "NO_DATA" in sql:
                return "No relevant data found in structured tables for this query."

            # ── Execute ────────────────────────────────────────────────────
            df = await self._pg.execute_sql(sql)

            # ── LOG 2: Execution result ────────────────────────────────────
            logger.info(
                "SQL executed successfully",
                generated_sql=sql,
                rows_returned=len(df),
                columns=list(df.columns),
            )

            if df.empty:
                return "The query returned no results."

            return self._format_result(df, query)

        except Exception as e:
            logger.error("NL→SQL execution failed", query=query, error=str(e))
            return f"Structured data lookup failed: {str(e)}"

    def _sanitize_sql(self, sql: str) -> str:
        """Remove markdown fences, validate it's a SELECT statement."""
        sql = re.sub(r"```sql\s*|\s*```", "", sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r"```\s*|\s*```", "", sql).strip()

        if not sql.upper().strip().startswith("SELECT"):
            raise ValueError(f"Non-SELECT statement generated: {sql[:100]}")

        # Disallow dangerous keywords even within SELECT
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE"]
        sql_upper = sql.upper()
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(f"Dangerous keyword '{keyword}' found in generated SQL")

        return sql

    def _format_result(self, df: pd.DataFrame, original_query: str) -> str:
        """Format DataFrame as readable context for the final LLM."""
        if len(df) == 1 and len(df.columns) == 1:
            # Single value result — e.g. COUNT(*), SUM(revenue)
            return f"Result: {df.iloc[0, 0]}"

        # Multi-row or multi-column result
        rows = df.head(20).to_string(index=False)
        return f"Query results ({len(df)} rows):\n\n{rows}"