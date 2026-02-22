"""
retrieval/nl_to_sql.py
───────────────────────
NL→SQL engine for the structured query path.
Uses LCEL: prompt | llm | StrOutputParser()
Executes against PostgreSQL with safety guardrails (SELECT only).
"""
from __future__ import annotations

import re
from typing import Optional

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
# NL→SQL Prompt — LCEL usage
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
    Only activated for queries classified as QueryType.STRUCTURED.
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
    async def execute(self, query: str, schema_context: str) -> str:
        """
        Generate SQL from natural language, execute it, return formatted result.
        Returns a string context suitable for passing to the final LLM.
        """
        try:
            # Generate SQL via LCEL
            sql = await self._chain.ainvoke({
                "query": query,
                "schema_context": schema_context,
            })

            # Safety: strip markdown if LLM wrapped in backticks
            sql = self._sanitize_sql(sql)
            logger.debug("Generated SQL", sql=sql[:200])

            if "NO_DATA" in sql:
                return "No relevant data found in structured tables for this query."

            # Execute
            df = await self._pg.execute_sql(sql)

            if df.empty:
                return "The query returned no results."

            # Format result as readable context
            return self._format_result(df, query)

        except Exception as e:
            logger.error("NL→SQL execution failed", query=query, error=str(e))
            return f"Structured data lookup failed: {str(e)}"

    def _sanitize_sql(self, sql: str) -> str:
        """Remove markdown, validate it's a SELECT statement."""
        # Strip markdown code fences
        sql = re.sub(r"```sql\s*|\s*```", "", sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r"```\s*|\s*```", "", sql).strip()

        # Safety: only allow SELECT statements
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
        """Format DataFrame as readable context for the LLM."""
        if len(df) == 1 and len(df.columns) == 1:
            # Single value result
            return f"Result: {df.iloc[0, 0]}"

        # Multi-row or multi-column result
        rows = df.head(20).to_string(index=False)
        return f"Query results ({len(df)} rows):\n\n{rows}"