"""
retrieval/query_classifier.py
──────────────────────────────
Query classifier + router — two responsibilities resolved in a single LLM call:

1. Namespace routing  → which Qdrant collections to search
2. Route decision     → SQL | VECTOR_STORE | BOTH + which exact tables

The schema registry is fetched live from PostgreSQL so the router always
knows every ingested CSV/XLSX table without any hardcoded configuration.
Uses LCEL: prompt | llm | JsonOutputParser()

UPDATED: build_schema_context now surfaces the LLM-generated description
prominently (before columns) so the classifier can match tables by *subject
matter* rather than only by column names.
"""
from __future__ import annotations

import json
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client.http.models import (
    FieldCondition, Filter, MatchAny, MatchValue, Range, MinShould,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Namespace, QueryPlan, QueryType, RouteDecision
from src.storage.sql.postgres_store import get_postgres_store

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Classification + Routing Prompt — LCEL
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query routing assistant for an enterprise RAG system.
Analyze the user query and return a single JSON object covering all routing decisions.
Return ONLY valid JSON — no explanation, no markdown.

Available namespaces (vector store collections):
{namespace_context}

Available SQL tables (structured data ingested from CSV/Excel files):
{schema_context}

Each SQL table entry is formatted as:
  Table `<table_name>`
  Description: <what the table is about — use this to judge relevance>
  Columns: <field names available for filtering/aggregation>
  Source: <original filename> | Rows: <row count>

How to select relevant_tables:
- Read the Description field first — it tells you the subject matter of the table.
- Only include tables whose Description clearly matches what the query is asking for.
- If no table's Description is relevant, return an empty relevant_tables list.

Vector store contains: PDF documents, PowerPoint slides, Word docs, Markdown files
(unstructured knowledge, reports, presentations, documentation)

Route destination rules:
- "SQL"          : query needs data lookup, aggregation, filtering, or calculation from tables
- "VECTOR_STORE" : query needs knowledge, explanations, or context from documents
- "BOTH"         : query needs data AND document context (e.g. "explain the revenue drop in Q3")

Query type rules:
- "semantic"    : prose / conceptual queries  → vector search
- "structured"  : numeric lookups, aggregations, specific data points → SQL
- "hybrid"      : needs both SQL data and vector context simultaneously

JSON schema:
{{
  "namespaces": ["<namespace value from the list above>"],
  "query_type": "semantic|structured|hybrid",
  "destination": "SQL|VECTOR_STORE|BOTH",
  "relevant_tables": ["<table_name from the SQL tables list — only tables whose Description matches>"],
  "reasoning": "<one sentence explaining the routing decision, mentioning which table descriptions matched>",
  "entities": ["<named entity>"],
  "keywords": ["<keyword>"],
  "intent": "<brief intent description>",
  "date_range_start": "YYYY-MM-DD or null",
  "date_range_end": "YYYY-MM-DD or null",
  "language": "en"
}}"""),
    ("human", "Query: {query}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Schema Context Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_context(schema_registry: dict) -> str:
    """
    Build a readable schema context string from the registry dict.
    Injected into the router prompt so the LLM can match tables by
    description (subject matter) first, then columns.

    Format per table:
        Table `tbl_abc123def456`
        Description: Monthly sales records for North America region covering 2022-2024.
        Columns: date, product_id, region, revenue, units_sold, discount_rate
        Source: sales_2024.csv | Rows: 1200

    The Description is placed first and prominently so the classifier reads
    it before trying to pattern-match column names.
    """
    if not schema_registry:
        return "No structured SQL tables available."

    blocks = []
    for table_name, info in schema_registry.items():
        cols = ", ".join(info.get("columns", []))
        source = info.get("source", "unknown")
        description = info.get("description", "").strip()
        row_count = info.get("row_count", 0)

        # Description line — show placeholder if somehow blank (fallback ingestion path)
        desc_line = description if description else f"Structured data from {source}."

        block = (
            f"Table `{table_name}`\n"
            f"  Description: {desc_line}\n"
            f"  Columns: {cols}\n"
            f"  Source: {source} | Rows: {row_count}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Prefilter Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prefilter(plan: QueryPlan, user_roles: List[str]) -> Filter:
    """
    Build a Qdrant pre-filter from the QueryPlan + user roles.
    Applied before ANN search so it doesn't degrade recall.

    Two tiers:
      must[]       — hard filters: chunk excluded if ANY of these fail.
                     Only factual, non-semantic conditions belong here.
      min_should[] — soft boost: optional conditions that improve precision
                     when matched but never exclude a chunk if unmatched.
                     min_count=0 means zero of these need to match —
                     they act as ranking hints, not hard gates.
    """
    must_conditions = []
    should_conditions = []

    # ── MUST: Access control ──────────────────────────────────────────────
    if user_roles:
        must_conditions.append(
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=user_roles),
            )
        )

    # ── MUST: Latest version only ─────────────────────────────────────────
    must_conditions.append(
        FieldCondition(key="is_latest", match=MatchValue(value=True))
    )

    # ── MUST: Date range — only when query explicitly targets a time period ─
    if plan.date_range_start or plan.date_range_end:
        date_range = {}
        if plan.date_range_start:
            date_range["gte"] = plan.date_range_start
        if plan.date_range_end:
            date_range["lte"] = plan.date_range_end
        must_conditions.append(
            FieldCondition(key="doc_date", range=Range(**date_range))
        )

    # ── SHOULD: Entity boost — soft signal, never a hard gate ────────────
    # Entities are LLM-extracted proper nouns (TechFlow, Salesforce, AWS).
    # They boost chunks mentioning the same entities as the query, but
    # min_count=0 means chunks without matching entities are still returned.
    # This preserves recall for queries like "mobile app" where no named
    # entity exists in the stored entity list.
    if plan.entities:
        should_conditions.append(
            FieldCondition(
                key="entities",
                match=MatchAny(any=plan.entities),
            )
        )

    min_should = (
        MinShould(conditions=should_conditions, min_count=0)
        if should_conditions
        else None
    )

    return Filter(
        must=must_conditions,
        min_should=min_should,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Classifies incoming queries for namespace routing and retrieval strategy.

    Fetches both namespace descriptions and the schema registry from DB on
    every call, so routing decisions are always current with what's been ingested.

    The schema registry now carries LLM-generated descriptions, so the
    classifier can match tables by subject matter rather than column names.
    Uses LCEL for clean, testable LLM integration.
    """

    def __init__(self):
        self._pg = get_postgres_store()
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        # LCEL chain: prompt → LLM → JSON
        self._chain = CLASSIFICATION_PROMPT | llm | JsonOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def classify(self, query: str, user_roles: List[str] = None) -> QueryPlan:
        """
        Classify a query and return a routing plan with embedded RouteDecision.
        A single LLM call resolves namespace, destination, and relevant_tables.
        Table selection is guided by LLM-generated descriptions in the schema registry.
        """
        try:
            # ── Fetch live context from DB ─────────────────────────────────
            namespaces_from_db = await self._pg.get_namespace_context()
            namespace_context  = json.dumps(namespaces_from_db, indent=2)

            schema_registry    = await self._pg.get_schema_registry()
            # build_schema_context now formats Description prominently
            schema_context     = build_schema_context(schema_registry)

            result = await self._chain.ainvoke({
                "query":             query,
                "namespace_context": namespace_context,
                "schema_context":    schema_context,
            })

            # ── Parse namespaces ───────────────────────────────────────────
            raw_namespaces = result.get("namespaces", ["GENERAL"])
            namespaces = []
            for ns_str in raw_namespaces:
                try:
                    ns = Namespace(ns_str)
                    if user_roles is None or self._can_access(ns, user_roles):
                        namespaces.append(ns)
                except ValueError:
                    pass
            if not namespaces:
                namespaces = [Namespace.GENERAL]

            # ── Parse query type ───────────────────────────────────────────
            raw_type = result.get("query_type", "semantic")
            try:
                query_type = QueryType(raw_type)
            except ValueError:
                query_type = QueryType.SEMANTIC

            # ── Parse route decision ───────────────────────────────────────
            raw_dest = result.get("destination", "VECTOR_STORE")
            valid_destinations = {"SQL", "VECTOR_STORE", "BOTH"}
            if raw_dest not in valid_destinations:
                raw_dest = "VECTOR_STORE"

            route_decision = RouteDecision(
                destination=raw_dest,
                reasoning=result.get("reasoning", ""),
                relevant_tables=result.get("relevant_tables", []),
            )

            plan = QueryPlan(
                original_query=query,
                namespaces=namespaces,
                query_type=query_type,
                entities=result.get("entities", []),
                keywords=result.get("keywords", []),
                date_range_start=result.get("date_range_start"),
                date_range_end=result.get("date_range_end"),
                intent=result.get("intent", ""),
                language=result.get("language", "en"),
                confidence=1.0,
                route_decision=route_decision,
            )

            logger.info(
                "Query classified",
                query=query[:80],
                namespaces=[ns.value for ns in namespaces],
                destination=raw_dest,
                relevant_tables=route_decision.relevant_tables,
                reasoning=route_decision.reasoning,
            )

            return plan

        except Exception as exc:
            logger.error("Query classification failed", query=query, error=str(exc))
            # Safe fallback — vector search on GENERAL namespace
            return QueryPlan(
                original_query=query,
                namespaces=[Namespace.GENERAL],
                query_type=QueryType.SEMANTIC,
                route_decision=RouteDecision(
                    destination="VECTOR_STORE",
                    reasoning="Classification failed — defaulting to vector search.",
                    relevant_tables=[],
                ),
            )

    @staticmethod
    def _can_access(namespace: Namespace, user_roles: List[str]) -> bool:
        """
        Simple role-based namespace gating.
        Extend this with a DB-backed ACL if needed.
        """
        # Currently all namespaces are accessible to any authenticated role.
        # Add namespace-specific restrictions here as needed.
        return True