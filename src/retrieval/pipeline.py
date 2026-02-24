"""
retrieval/pipeline.py
──────────────────────
The retrieval pipeline — orchestrates all retrieval stages:

1. Query classification & routing (namespace + SQL/VECTOR/BOTH decision)
2. Route execution:
   - SQL path         → NL→SQL → execute → generate answer
   - Vector path      → hybrid search → context expansion → rerank → generate
   - Hybrid path      → SQL + vector concurrently → merge contexts → generate
3. Caching (Redis) — keyed by (query, sorted user_roles)
4. Query logging

RouteDecision (destination: SQL | VECTOR_STORE | BOTH) is produced by the
QueryClassifier in a single LLM call and drives which path(s) execute here.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import (
    Chunk, Namespace, QueryPlan, QueryType,
    RAGResponse, RetrievedChunk,
)
from src.retrieval.nl_to_sql import NLToSQLEngine
from src.retrieval.query_classifier import QueryClassifier, build_prefilter
from src.retrieval.rerankers.bge_reranker import get_reranker
from src.storage.cache.redis_store import get_redis_cache
from src.storage.sql.postgres_store import get_postgres_store
from src.storage.vector.embedder import get_embedder
from src.storage.vector.qdrant_store import get_qdrant_store

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generation Prompt — LCEL
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful enterprise assistant. Answer the user's question
using ONLY the provided context. Be precise and factual.

Rules:
- If the context doesn't contain the answer, say "I don't have relevant information to answer this."
- Never fabricate information not present in the context
- Cite the source section when helpful (e.g., "According to the HR Policies document...")
- Be concise but complete

Context:
{context}"""),
    ("human", "{question}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_schema_context_string(schema_registry: Dict) -> str:
    """
    Format the schema_registry dict as a readable string for SQL prompts.
    Used when no relevant_tables are specified (full context fallback).
    """
    if not schema_registry:
        return "No structured tables available."
    lines = []
    for table_name, info in schema_registry.items():
        cols = ", ".join(info.get("columns", []))
        lines.append(
            f"Table `{table_name}`: columns=[{cols}] | source={info['source']} "
            f"| rows={info['row_count']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Orchestrates the complete retrieval pipeline.
    Manages: caching, namespace routing, hybrid search,
    context expansion, reranking, SQL execution, and answer generation.
    """

    def __init__(self):
        self.classifier = QueryClassifier()
        self.embedder   = get_embedder()
        self.qdrant     = get_qdrant_store()
        self.reranker   = get_reranker()
        self.cache      = get_redis_cache()
        self.pg         = get_postgres_store()
        self.nl_to_sql  = NLToSQLEngine()

        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        # LCEL answer generation chain
        self._answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    async def query(
        self,
        user_query: str,
        user_roles: Optional[List[str]] = None,
        top_k: int = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: classify → route → retrieve → rerank → generate.
        """
        start_time = time.time()
        user_roles = user_roles or ["EMPLOYEE"]
        top_k = top_k or settings.retrieval_top_k

        # ── Stage 0: Cache Check ──────────────────────────────────────────────
        # Cache is keyed by (query, user_roles) — roles scope the cache for security.
        cached = await self.cache.get_cached_response(
            user_query,
            sorted(user_roles),
        )
        if cached:
            logger.info("Query served from cache", query=user_query[:60])
            return RAGResponse(
                answer=cached["answer"],
                source_chunks=[],
                query_plan=QueryPlan(
                    original_query=user_query,
                    namespaces=[Namespace(ns) for ns in cached.get("namespaces", ["GENERAL"])],
                    query_type=QueryType(cached.get("query_type", "semantic")),
                ),
                latency_ms=(time.time() - start_time) * 1000,
                cached=True,
            )

        # ── Stage 1: Query Classification + Routing ───────────────────────────
        plan = await self.classifier.classify(user_query, user_roles)

        destination = plan.route_decision.destination if plan.route_decision else "VECTOR_STORE"
        logger.info(
            "Query classified",
            query=user_query[:60],
            destination=destination,
            reasoning=plan.route_decision.reasoning if plan.route_decision else "",
            relevant_tables=plan.route_decision.relevant_tables if plan.route_decision else [],
            namespaces=[ns.value for ns in plan.namespaces],
            query_type=plan.query_type.value,
        )

        # ── Stage 2: Route by destination ─────────────────────────────────────
        if destination == "SQL":
            answer, chunks = await self._structured_path(user_query, plan)
        elif destination == "BOTH":
            answer, chunks = await self._hybrid_path(user_query, plan, user_roles, top_k)
        else:
            # VECTOR_STORE (default)
            answer, chunks = await self._semantic_path(user_query, plan, user_roles, top_k)

        # ── Answer Generation (semantic path returns chunks, not answer) ───────
        if not answer:
            answer = await self._generate_answer(user_query, chunks)

        latency_ms = (time.time() - start_time) * 1000

        # ── Cache the response ────────────────────────────────────────────────
        if answer and "I don't have relevant information" not in answer:
            await self.cache.cache_response(
                user_query,
                sorted(user_roles),
                {
                    "answer":     answer,
                    "query_type": plan.query_type.value,
                    "namespaces": [ns.value for ns in plan.namespaces],
                },
            )

        # ── Log query ─────────────────────────────────────────────────────────
        await self.pg.log_query(
            query_text=user_query,
            query_type=plan.query_type.value,
            namespaces=[ns.value for ns in plan.namespaces],
            latency_ms=latency_ms,
            chunk_count=len(chunks),
            cached=False,
        )

        return RAGResponse(
            answer=answer,
            source_chunks=chunks,
            query_plan=plan,
            latency_ms=latency_ms,
            cached=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Semantic Path (vector retrieval)
    # ─────────────────────────────────────────────────────────────────────────

    async def _semantic_path(
        self,
        query: str,
        plan: QueryPlan,
        user_roles: List[str],
        top_k: int,
    ):
        """Vector-only path: hybrid search → expand → rerank."""
        embedding = self.embedder.embed_text(query)
        prefilter = build_prefilter(plan, user_roles)

        # Search across all target namespaces (parallel for multi-namespace)
        all_candidates: List[RetrievedChunk] = []
        for namespace in plan.namespaces:
            candidates = self.qdrant.hybrid_search(
                dense_vector=embedding.dense,
                sparse_vector=embedding.sparse,
                namespace=namespace,
                prefilter=prefilter,
                top_k=top_k,
            )
            all_candidates.extend(candidates)

        # Sort by score and deduplicate
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        all_candidates = self._deduplicate(all_candidates)[:top_k]

        # Apply minimum similarity threshold
        above_threshold = [c for c in all_candidates if c.score >= settings.min_similarity_score]
        if not above_threshold:
            logger.info("No chunks above similarity threshold", query=query[:60])
            return "I don't have relevant information to answer this question.", []

        # Stage 3: Context Expansion (child → parent)
        expanded = self._expand_to_parents(above_threshold, plan.namespaces[0])

        # Stage 4: Rerank
        reranked = self.reranker.rerank(query, expanded, top_n=settings.reranker_top_n)
        reranked = [r for r in reranked if r.score > 0.05]
        seen = set()
        reranked = [r for r in reranked if not (r.chunk.chunk_id in seen or seen.add(r.chunk.chunk_id))]

        return None, reranked   # Answer generated separately by _generate_answer

    # ─────────────────────────────────────────────────────────────────────────
    # Structured Path (NL→SQL)
    # ─────────────────────────────────────────────────────────────────────────

    async def _structured_path(self, query: str, plan: QueryPlan):
        """
        SQL-only path: fetch schema registry → build focused context →
        NL→SQL → execute → generate answer.
        """
        schema_registry = await self.pg.get_schema_registry()
        schema_context  = _build_schema_context_string(schema_registry)

        relevant_tables = (
            plan.route_decision.relevant_tables
            if plan.route_decision else []
        )

        sql_result = await self.nl_to_sql.execute(
            query,
            schema_context,
            relevant_tables=relevant_tables,
        )

        final_answer = await self._answer_chain.ainvoke({
            "context":  f"--- Data from SQL ---\n{sql_result}",
            "question": query,
        })
        return final_answer, []

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid Path (SQL + Vector concurrently)
    # ─────────────────────────────────────────────────────────────────────────

    async def _hybrid_path(
        self,
        query: str,
        plan: QueryPlan,
        user_roles: List[str],
        top_k: int,
    ):
        """
        BOTH path: run SQL and vector search concurrently, merge contexts,
        generate a single answer from the combined result.

        SQL gets the router-selected tables for focused querying.
        Vector gets the full hybrid search + rerank pipeline.
        Context tokens are split 50/50 between SQL and vector results.
        """
        # Run both paths concurrently
        sql_task    = asyncio.create_task(self._structured_path(query, plan))
        vector_task = asyncio.create_task(self._semantic_path(query, plan, user_roles, top_k))

        (sql_answer, _), (_, vector_chunks) = await asyncio.gather(sql_task, vector_task)

        # ── Build merged context ───────────────────────────────────────────
        context_parts = []

        if sql_answer and "No relevant data" not in sql_answer and "I don't have" not in sql_answer:
            context_parts.append(f"--- Data from SQL ---\n{sql_answer}")

        if vector_chunks:
            raw_chunks = []
            total_tokens = 0
            token_budget = settings.max_context_tokens // 2   # share budget with SQL
            for rc in vector_chunks:
                chunk_tokens = rc.chunk.token_count or len(rc.chunk.text) // 4
                if total_tokens + chunk_tokens > token_budget:
                    break
                source = rc.chunk.section_heading or rc.chunk.source_url or "Document"
                raw_chunks.append(f"[Source: {source}]\n{rc.chunk.text}")
                total_tokens += chunk_tokens
            if raw_chunks:
                context_parts.append(
                    "--- Context from Documents ---\n" + "\n\n".join(raw_chunks)
                )

        combined_context = "\n\n".join(context_parts) if context_parts else \
            "No relevant context found."

        final_answer = await self._answer_chain.ainvoke({
            "context":  combined_context,
            "question": query,
        })

        logger.info(
            "Hybrid path complete",
            query=query[:60],
            sql_used=bool(sql_answer and "No relevant data" not in sql_answer),
            vector_chunks=len(vector_chunks),
        )

        return final_answer, vector_chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Answer Generation (LCEL) — used by semantic path
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I don't have relevant information to answer this question."

        # Build context string — cap at max_context_tokens
        context_parts = []
        total_tokens = 0
        for rc in chunks:
            chunk_tokens = rc.chunk.token_count or len(rc.chunk.text) // 4
            if total_tokens + chunk_tokens > settings.max_context_tokens:
                break
            source = rc.chunk.section_heading or rc.chunk.source_url or "Document"
            context_parts.append(f"[Source: {source}]\n{rc.chunk.text}")
            total_tokens += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)

        answer = await self._answer_chain.ainvoke({
            "context":  context,
            "question": query,
        })
        return answer

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _expand_to_parents(
        self,
        chunks: List[RetrievedChunk],
        namespace: Namespace,
    ) -> List[RetrievedChunk]:
        """
        Expand child chunks to their parent sections for richer context.
        Deduplicates parents (multiple children may share the same parent).
        """
        expanded = []
        seen_parent_ids = set()

        for rc in chunks:
            parent_id = rc.chunk.parent_id
            if parent_id and parent_id not in seen_parent_ids:
                parent = self.qdrant.get_chunk_by_id(parent_id, namespace)
                if parent:
                    expanded.append(RetrievedChunk(
                        chunk=parent,
                        score=rc.score,
                        source="vector_expanded",
                    ))
                    seen_parent_ids.add(parent_id)
                else:
                    expanded.append(rc)
            elif not parent_id:
                expanded.append(rc)

        return expanded

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate chunks by content hash."""
        seen = set()
        result = []
        for rc in chunks:
            h = rc.chunk.content_hash
            if h not in seen:
                seen.add(h)
                result.append(rc)
        return result

    async def _format_schema_context(self, schemas: List[dict]) -> str:
        """Legacy helper — still used if called directly with tabular_datasets rows."""
        if not schemas:
            return "No tabular data available."
        parts = []
        for s in schemas:
            cols = ", ".join(f"{k} ({v})" for k, v in s["column_schema"].items())
            limit = s["row_count"] if s["row_count"] <= 20 else 5
            sample_rows = await self._get_sample_rows(table_name=s["table_name"], n=limit)
            parts.append(
                f"Table: {s['table_name']} (from {s['sheet_name']})\n"
                f"Columns: {cols}\n"
                f"Row count: {s['row_count']}\n"
                f"Sample data:\n{sample_rows}"
            )
        return "\n\n".join(parts)

    async def _get_sample_rows(self, table_name: str, n: int = 5) -> str:
        try:
            df = await self.pg.execute_sql(f'SELECT * FROM "{table_name}" LIMIT {n}')
            return df.to_string(index=False)
        except Exception as e:
            logger.warning("Failed to fetch sample rows", table=table_name, error=str(e))
            return "Sample data unavailable"