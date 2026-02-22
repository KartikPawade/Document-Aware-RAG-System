"""
retrieval/rerankers/bge_reranker.py
─────────────────────────────────────
BGE-Reranker-v2-m3 cross-encoder reranker — fully self-hosted.
Takes top-K candidates from hybrid search and re-scores them jointly
with the query for higher precision.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


class BGEReranker:
    """
    Wraps BAAI/bge-reranker-v2-m3 from FlagEmbedding.
    Cross-encoder: scores (query, document) pairs jointly.
    Much higher precision than bi-encoder ANN search.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                logger.info("Loading BGE Reranker", model=settings.reranker_model_name)
                self._model = FlagReranker(
                    settings.reranker_model_name,
                    use_fp16=True,
                    device=settings.embedding_device,
                )
                logger.info("BGE Reranker loaded")
            except ImportError:
                logger.warning("FlagEmbedding not available, using sentence-transformers reranker")
                from sentence_transformers import CrossEncoder
                self._model = _CrossEncoderWrapper(settings.reranker_model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank candidate chunks by relevance to query.
        Returns top_n chunks sorted by reranker score (descending).
        """
        if not candidates:
            return []

        top_n = top_n or settings.reranker_top_n
        model = self._get_model()

        # Build (query, document) pairs
        pairs = [(query, chunk.chunk.text) for chunk in candidates]

        try:
            scores: List[float] = model.compute_score(
                pairs,
                batch_size=settings.reranker_batch_size,
                normalize=True,          # Normalize to [0, 1]
            )

            # Attach reranker scores and sort
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            result = []
            for chunk, score in scored[:top_n]:
                chunk.score = float(score)
                result.append(chunk)

            logger.debug(
                "Reranking complete",
                candidates=len(candidates),
                returned=len(result),
                top_score=result[0].score if result else 0,
            )
            return result

        except Exception as e:
            logger.warning("Reranking failed, returning original order", error=str(e))
            return candidates[:top_n]


class _CrossEncoderWrapper:
    """Fallback: sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = CrossEncoder(fallback)

    def compute_score(self, pairs, batch_size=16, normalize=True) -> List[float]:
        scores = self._model.predict(pairs, batch_size=batch_size)
        if normalize:
            import numpy as np
            scores = 1 / (1 + np.exp(-scores))  # Sigmoid normalization
        return scores.tolist()


# Singleton
_reranker: Optional[BGEReranker] = None

def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker