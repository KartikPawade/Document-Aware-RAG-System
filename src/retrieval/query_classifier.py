"""
retrieval/query_classifier.py
──────────────────────────────
Query classifier + router — two responsibilities resolved in a single LLM call:

1. Namespace routing  → which Qdrant collections to search
2. Route decision     → SQL | VECTOR_STORE | BOTH + which exact tables

The schema registry is fetched live from PostgreSQL so the router always
knows every ingested CSV/XLSX table without any hardcoded configuration.
Uses LCEL: prompt | llm | JsonOutputParser()
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
  "relevant_tables": ["<table_name from the SQL tables list>"],
  "reasoning": "<one sentence explaining the routing decision>",
  "entities": ["<named entity>"],
  "keywords": ["<keyword>"],
  "intent": "<brief intent description>",
  "date_range_start": "YYYY-MM-DD or null",
  "date_range_end": "YYYY-MM-DD or null",
  "language": "en"
}}"""),
    ("human", "Query: {query}"),
])


def build_schema_context(schema_registry: dict) -> str:
    """
    Build a readable schema context string from the registry dict.
    Injected into the router prompt so the LLM knows every available SQL table.

    Example output line:
        Table `tbl_abc123`: columns=[date, product, revenue] | source=sales_2024.csv
                           | rows=1200 | sales_2024.csv — sheet 'default'. Contains 1200 rows...
    """
    if not schema_registry:
        return "No structured SQL tables available."

    lines = []
    for table_name, info in schema_registry.items():
        cols = ", ".join(info.get("columns", []))
        source = info.get("source", "")
        desc = info.get("description", "")
        row_count = info.get("row_count", 0)
        lines.append(
            f"Table `{table_name}`: columns=[{cols}] | source={source} "
            f"| rows={row_count} | {desc}"
        )
    return "\n".join(lines)


class QueryClassifier:
    """
    Classifies incoming queries for namespace routing and retrieval strategy.

    Fetches both namespace descriptions and the schema registry from DB on
    every call, so routing decisions are always current with what's been ingested.
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
        """
        try:
            # ── Fetch live context from DB ─────────────────────────────────
            namespaces_from_db = await self._pg.get_namespace_context()
            namespace_context  = json.dumps(namespaces_from_db, indent=2)

            schema_registry    = await self._pg.get_schema_registry()
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
            qt_map = {
                "structured": QueryType.STRUCTURED,
                "hybrid":     QueryType.HYBRID,
            }
            query_type = qt_map.get(result.get("query_type", "semantic"), QueryType.SEMANTIC)

            # ── Parse route decision ───────────────────────────────────────
            raw_destination = result.get("destination", "VECTOR_STORE")

            # Normalise: BOTH always forces HYBRID query type
            if raw_destination == "BOTH":
                query_type = QueryType.HYBRID

            route_decision = RouteDecision(
                destination=raw_destination,
                reasoning=result.get("reasoning", ""),
                relevant_tables=result.get("relevant_tables", []),
            )

            logger.info(
                "Query classified",
                query=query[:60],
                destination=route_decision.destination,
                reasoning=route_decision.reasoning,
                relevant_tables=route_decision.relevant_tables,
                namespaces=[ns.value for ns in namespaces],
                query_type=query_type.value,
            )

            return QueryPlan(
                original_query=query,
                namespaces=namespaces,
                query_type=query_type,
                entities=result.get("entities", []),
                keywords=result.get("keywords", []),
                date_range_start=result.get("date_range_start"),
                date_range_end=result.get("date_range_end"),
                intent=result.get("intent", ""),
                language=result.get("language", "en"),
                route_decision=route_decision,
            )

        except Exception as e:
            logger.warning("Query classification failed, using defaults", error=str(e))
            return QueryPlan(
                original_query=query,
                namespaces=[Namespace.GENERAL],
                query_type=QueryType.SEMANTIC,
                route_decision=RouteDecision(
                    destination="VECTOR_STORE",
                    reasoning="Classification failed — defaulting to vector search",
                    relevant_tables=[],
                ),
            )

    def _can_access(self, namespace: Namespace, roles: List[str]) -> bool:
        """
        Access control: check if user roles allow access to namespace.
        Extend this mapping to match your RBAC system.
        """
        restricted = {
            Namespace.HR_EMPLOYEES: ["HR", "MANAGER", "ADMIN"],
            Namespace.FINANCE:      ["FINANCE", "ADMIN"],
            Namespace.LEGAL:        ["LEGAL", "ADMIN"],
        }
        allowed = restricted.get(namespace)
        if allowed is None:
            return True   # Not restricted — all roles can access
        return bool(set(roles) & set(allowed))


def build_prefilter(plan: QueryPlan, user_roles: List[str] = None) -> Filter | None:
    """
    Build a Qdrant pre-filter from the query plan.
    Applied before ANN search for access control and date scoping.
    """
    conditions = []

    if plan.entities:
        conditions.append(
            FieldCondition(key="entities", match=MatchAny(any=plan.entities))
        )
    if plan.date_range_start or plan.date_range_end:
        conditions.append(
            FieldCondition(
                key="doc_date",
                range=Range(
                    gte=plan.date_range_start,
                    lte=plan.date_range_end,
                ),
            )
        )
    if user_roles:
        conditions.append(
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=user_roles),
            )
        )

    return Filter(must=conditions) if conditions else None