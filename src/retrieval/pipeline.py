"""
retrieval/pipeline.py
──────────────────────
The retrieval pipeline — orchestrates all 5 retrieval stages:
1. Query classification & planning
2. Hybrid search (dense + sparse + RRF)
3. Small-to-large context expansion (parent chunks)
4. Cross-encoder reranking (BGE-Reranker)
5. SQL path for structured queries
"""
from __future__ import annotations

import time
from typing import List, Optional

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


class RetrievalPipeline:
    """
    Orchestrates the complete retrieval pipeline.
    Manages: caching, namespace routing, hybrid search,
    context expansion, reranking, and answer generation.
    """

    def __init__(self):
        self.classifier   = QueryClassifier()
        self.embedder     = get_embedder()
        self.qdrant       = get_qdrant_store()
        self.reranker     = get_reranker()
        self.cache        = get_redis_cache()
        self.pg           = get_postgres_store()
        self.nl_to_sql    = NLToSQLEngine()

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
        Full RAG pipeline: classify → retrieve → rerank → generate.
        """
        start_time = time.time()
        user_roles = user_roles or ["EMPLOYEE"]
        top_k = top_k or settings.retrieval_top_k

        # ── Stage 0: Cache Check ──────────────────────────────────────────────
        # Cache is keyed by (query, user_roles) — roles scope the cache for security.
        # Same query from same role set hits cache; different roles get separate entries.
        cached = await self.cache.get_cached_response(
            user_query,
            sorted(user_roles),  # Sort for deterministic key regardless of role order
        )
        if cached:
            logger.info("Query served from cache", query=user_query[:60])
            return RAGResponse(
                answer=cached["answer"],
                source_chunks=[],
                query_plan=QueryPlan(
                    original_query=user_query,
                    namespaces=[Namespace.GENERAL],
                    query_type=QueryType.SEMANTIC,
                ),
                latency_ms=(time.time() - start_time) * 1000,
                cached=True,
            )

        # ── Stage 1: Query Classification ─────────────────────────────────────
        plan = await self.classifier.classify(user_query, user_roles)
        logger.info(
            "Query classified",
            query=user_query[:60],
            namespaces=[ns.value for ns in plan.namespaces],
            query_type=plan.query_type,
        )

        # ── Stage 2: Route by Query Type ──────────────────────────────────────
        if plan.query_type == QueryType.STRUCTURED:
            answer, chunks = await self._structured_path(user_query, plan)
        else:
            answer, chunks = await self._semantic_path(user_query, plan, user_roles, top_k)

        # ── Answer Generation (semantic path returns chunks, not answer) ───────
        if not answer:
            answer = await self._generate_answer(user_query, chunks)

        latency_ms = (time.time() - start_time) * 1000

        # ── Cache the response ────────────────────────────────────────────────
        # Only cache if we got a real answer
        if answer and "I don't have relevant information" not in answer:
            await self.cache.cache_response(
            user_query,
            sorted(user_roles),
            {"answer": answer},
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
        # Embed the query
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

        reranked = [r for r in reranked if r.score > 0.05]  # remove irrelevant chunks
        seen = set()
        reranked = [r for r in reranked if not (r.chunk.chunk_id in seen or seen.add(r.chunk.chunk_id))]  # remove duplicates


        return None, reranked   # Answer generated separately

    # ─────────────────────────────────────────────────────────────────────────
    # Structured Path (NL→SQL)
    # ─────────────────────────────────────────────────────────────────────────

    async def _structured_path(self, query: str, plan: QueryPlan):
        # Get table schemas for context
        schemas = await self.pg.get_table_schema(source_id="")
        schema_context = self._format_schema_context(schemas)

        # Execute NL→SQL
        sql_result = await self.nl_to_sql.execute(query, schema_context)

        # Generate final answer from SQL result : pass sql_result as string context, not raw
        final_answer = await self._answer_chain.ainvoke({
            "context": str(sql_result),
            "question": query,
        })
        return final_answer, []

    # ─────────────────────────────────────────────────────────────────────────
    # Answer Generation (LCEL)
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

        # LCEL answer generation
        answer = await self._answer_chain.ainvoke({
            "context": context,
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

    def _format_schema_context(self, schemas: List[dict]) -> str:
        if not schemas:
            return "No tabular data available."
        parts = []
        for s in schemas:
            cols = ", ".join(f"{k} ({v})" for k, v in s["column_schema"].items())
            parts.append(
                f"Table: {s['table_name']} (from {s['sheet_name']})\n"
                f"Columns: {cols}\n"
                f"Row count: {s['row_count']}"
            )
        return "\n\n".join(parts)