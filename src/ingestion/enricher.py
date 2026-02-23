"""
ingestion/enricher.py
──────────────────────
LLM-based metadata enrichment for chunks.
Uses LCEL (prompt | llm | parser) — appropriate use of LangChain.
Adds: domain, entities, keywords, summary, hypothetical_questions.
Falls back to spaCy NER if LLM call fails.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Chunk

logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Enrichment Prompt
# ─────────────────────────────────────────────────────────────────────────────

ENRICHMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a metadata extraction assistant for an enterprise RAG system.
Analyze the provided document chunk and return a JSON object with exactly these fields.
Return ONLY valid JSON — no explanation, no markdown, no extra text.


JSON schema:
{{ 
  "domain": "<business domain string>",
  "subdomain": "<specific subdomain>",
  "entities": ["<named entity>"],
  "keywords": ["<keyword>"],
  "summary": "<1-2 sentence description>",
  "hypothetical_questions": ["<question this chunk answers>"],
  "language": "<ISO 639-1 language code>"
}}"""),
    ("human", "Chunk to analyze:\n\n{chunk_text}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Enricher
# ─────────────────────────────────────────────────────────────────────────────

class MetadataEnricher:
    """
    Enriches chunks with LLM-generated metadata.
    Uses LCEL: prompt | llm | JsonOutputParser()

    Falls back to spaCy NER if LLM fails — ingestion is never blocked.
    """

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._chain = None
        self._nlp = None  # spaCy fallback, lazy-loaded

    def _get_chain(self):
        if self._chain is None:
            self._llm = ChatOpenAI(
                model=settings.openai_enrichment_model,
                temperature=0.0,
                api_key=settings.openai_api_key,
            )
            # LCEL: prompt → llm → json parser
            self._chain = ENRICHMENT_PROMPT | self._llm | JsonOutputParser()
        return self._chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """Enrich a single chunk with LLM-generated metadata."""
        try:
            chain = self._get_chain()
            result: Dict[str, Any] = await chain.ainvoke({"chunk_text": chunk.text})
            self._apply_enrichment(chunk, result)
            logger.debug("Chunk enriched", chunk_id=chunk.chunk_id, namespace=chunk.namespace)

        except Exception as e:
            logger.warning(
                "LLM enrichment failed, falling back to spaCy",
                chunk_id=chunk.chunk_id,
                error=str(e),
            )
            self._spacy_fallback(chunk)

        return chunk

    async def enrich_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enrich a batch of chunks concurrently."""
        import asyncio
        tasks = [self.enrich_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _apply_enrichment(self, chunk: Chunk, result: Dict[str, Any]) -> None:
        """Apply LLM enrichment result to chunk metadata."""        

        chunk.domain    = result.get("domain", "")
        chunk.subdomain = result.get("subdomain", "")
        chunk.entities  = result.get("entities", [])
        chunk.keywords  = result.get("keywords", [])
        chunk.summary   = result.get("summary", "")
        chunk.language  = result.get("language", "en")

        # Hypothetical questions — the HyDE-style enrichment
        chunk.hypothetical_questions = result.get("hypothetical_questions", [])

    def _spacy_fallback(self, chunk: Chunk) -> None:
        """Fallback: use spaCy for basic entity extraction."""
        try:
            if self._nlp is None:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            doc = self._nlp(chunk.text[:1000])  # Limit for speed
            chunk.entities = list({ent.text for ent in doc.ents})[:10]
            chunk.keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ][:15]
        except Exception as e:
            logger.warning("spaCy fallback also failed", error=str(e))
            # Last resort: empty enrichment (ingestion continues)