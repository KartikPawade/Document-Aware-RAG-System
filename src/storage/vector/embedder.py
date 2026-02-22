"""
storage/vector/embedder.py
───────────────────────────
BGE-M3 embedding service — self-hosted, no external API.
BGE-M3 produces dense + sparse vectors from a single model.
This eliminates the need for a separate BM25 service.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import numpy as np

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.models import Chunk

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingResult(NamedTuple):
    dense: List[float]               # Dense vector (1024-dim for BGE-M3)
    sparse: Dict[int, float]         # Sparse vector {token_id: weight}
    token_count: int


class BGEEmbedder:
    """
    Wraps BAAI/bge-m3 from FlagEmbedding.
    Produces dense and sparse (lexical weights) vectors in one pass.
    Lazy-loaded on first use to avoid startup delay.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                logger.info("Loading BGE-M3 model", model=settings.embedding_model_name)
                self._model = BGEM3FlagModel(
                    settings.embedding_model_name,
                    use_fp16=True,              # Faster inference
                    device=settings.embedding_device,
                )
                logger.info("BGE-M3 model loaded")
            except ImportError:
                logger.warning("FlagEmbedding not available, using sentence-transformers fallback")
                from sentence_transformers import SentenceTransformer
                self._model = _SentenceTransformerWrapper(settings.embedding_model_name)
        return self._model

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Embed a batch of texts.
        Returns dense + sparse vectors for each text.
        """
        model = self._get_model()
        results = []

        # Process in sub-batches
        batch_size = settings.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                output = model.encode(
                    batch,
                    batch_size=batch_size,
                    max_length=8192,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                for j in range(len(batch)):
                    dense = output["dense_vecs"][j].tolist()
                    sparse_weights = output.get("lexical_weights", [{}])[j]
                    # Convert to {int: float} format
                    sparse = {int(k): float(v) for k, v in sparse_weights.items()}
                    results.append(EmbeddingResult(
                        dense=dense,
                        sparse=sparse,
                        token_count=len(batch[j].split()),
                    ))
            except Exception as e:
                logger.error("Embedding batch failed", error=str(e), batch_size=len(batch))
                # Return zero vectors as fallback to not block pipeline
                dim = settings.dense_vector_size
                for _ in batch:
                    results.append(EmbeddingResult(
                        dense=[0.0] * dim,
                        sparse={},
                        token_count=0,
                    ))

        return results

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """Embed a list of Chunk objects."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_batch(texts)


class _SentenceTransformerWrapper:
    """
    Fallback wrapper around sentence-transformers.
    Produces dense vectors only — sparse will be empty.
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        dense = self._model.encode(texts, convert_to_numpy=True)
        return {
            "dense_vecs": dense,
            "lexical_weights": [{} for _ in texts],
        }


# Singleton
_embedder: Optional[BGEEmbedder] = None

def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder