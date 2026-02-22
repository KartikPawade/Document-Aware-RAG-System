"""
retrieval/query_classifier.py
──────────────────────────────
Query classifier: determines namespace(s), query type (semantic vs SQL),
entities, and date filters from a natural language query.
Uses LCEL: prompt | llm | JsonOutputParser()
"""
from __future__ import annotations

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client.http.models import (
    FieldCondition, Filter, MatchAny, MatchValue, Range,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Namespace, QueryPlan, QueryType

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Classification Prompt — LCEL usage
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query routing assistant for an enterprise RAG system.
Analyze the user query and return a JSON object for routing decisions.
Return ONLY valid JSON — no explanation, no markdown.

Available namespaces:
- HR_EMPLOYEES: employee profiles, org charts, roles, performance reviews
- HR_POLICIES: company policies, handbooks, PTO rules, code of conduct
- FINANCE: budgets, forecasts, invoices, financial reports
- TECH_DOCS: API docs, architecture, runbooks, code documentation
- LEGAL: contracts, NDAs, compliance, regulatory documents
- PRODUCTS: product specs, roadmaps, release notes, user guides
- GENERAL: anything that doesn't fit the above

Query types:
- "semantic": prose/conceptual queries → vector search
- "structured": numeric lookups, aggregations, specific data points → SQL

JSON schema:
{{
  "namespaces": ["<namespace>"],
  "query_type": "semantic|structured",
  "entities": ["<named entity>"],
  "keywords": ["<keyword>"],
  "intent": "<brief intent description>",
  "date_range_start": "YYYY-MM-DD or null",
  "date_range_end": "YYYY-MM-DD or null",
  "language": "en"
}}"""),
    ("human", "Query: {query}"),
])


class QueryClassifier:
    """
    Classifies incoming queries for namespace routing and retrieval strategy.
    Uses LCEL for clean, testable LLM integration.
    """

    def __init__(self):
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        # LCEL chain
        self._chain = CLASSIFICATION_PROMPT | llm | JsonOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def classify(self, query: str, user_roles: List[str] = None) -> QueryPlan:
        """Classify a query and return a routing plan."""
        try:
            result = await self._chain.ainvoke({"query": query})

            # Parse namespaces
            raw_namespaces = result.get("namespaces", ["GENERAL"])
            namespaces = []
            for ns_str in raw_namespaces:
                try:
                    ns = Namespace(ns_str)
                    # Enforce access control — only include namespaces user can access
                    if user_roles is None or self._can_access(ns, user_roles):
                        namespaces.append(ns)
                except ValueError:
                    pass

            if not namespaces:
                namespaces = [Namespace.GENERAL]

            # Parse query type
            qt_str = result.get("query_type", "semantic")
            query_type = QueryType.STRUCTURED if qt_str == "structured" else QueryType.SEMANTIC

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
            )

        except Exception as e:
            logger.warning("Query classification failed, using defaults", error=str(e))
            return QueryPlan(
                original_query=query,
                namespaces=[Namespace.GENERAL],
                query_type=QueryType.SEMANTIC,
            )

    def _can_access(self, namespace: Namespace, roles: List[str]) -> bool:
        """
        Access control: check if user roles allow access to namespace.
        Extend this mapping to match your RBAC system.
        """
        namespace_roles = {
            Namespace.HR_EMPLOYEES:  {"HR_MANAGER", "EXEC", "ADMIN"},
            Namespace.HR_POLICIES:   {"EMPLOYEE", "HR_MANAGER", "EXEC", "ADMIN"},
            Namespace.FINANCE:       {"FINANCE_ANALYST", "EXEC", "ADMIN"},
            Namespace.TECH_DOCS:     {"EMPLOYEE", "ENGINEER", "EXEC", "ADMIN"},
            Namespace.LEGAL:         {"LEGAL", "EXEC", "ADMIN"},
            Namespace.PRODUCTS:      {"EMPLOYEE", "EXEC", "ADMIN"},
            Namespace.GENERAL:       {"EMPLOYEE", "EXEC", "ADMIN"},
        }
        allowed = namespace_roles.get(namespace, set())
        return bool(set(roles) & allowed)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-filter Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prefilter(plan: QueryPlan, user_roles: List[str] = None) -> Filter:
    """
    Build Qdrant pre-filter from a QueryPlan.
    Applied BEFORE ANN search — collapses the search space.
    """
    conditions = []

    # Always: only latest chunk versions
    conditions.append(
        FieldCondition(key="is_latest", match=MatchValue(value=True))
    )

    # Always: namespace isolation (enforced by collection selection, but belt+suspenders)
    conditions.append(
        FieldCondition(key="namespace", match=MatchAny(any=[ns.value for ns in plan.namespaces]))
    )

    # Access control — hard filter
    if user_roles:
        conditions.append(
            FieldCondition(key="access_roles", match=MatchAny(any=user_roles))
        )

    # Entity targeting — dramatically reduces search space for specific entity queries
    if plan.entities:
        conditions.append(
            FieldCondition(key="entities", match=MatchAny(any=plan.entities))
        )

    # Language filter
    if plan.language and plan.language != "en":
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=plan.language))
        )

    # Temporal filters
    if plan.date_range_start or plan.date_range_end:
        range_filter = {}
        if plan.date_range_start:
            range_filter["gte"] = plan.date_range_start
        if plan.date_range_end:
            range_filter["lte"] = plan.date_range_end
        conditions.append(
            FieldCondition(key="doc_date", range=Range(**range_filter))
        )

    return Filter(must=conditions)