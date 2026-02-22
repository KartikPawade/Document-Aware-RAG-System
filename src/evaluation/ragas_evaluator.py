"""
evaluation/ragas_evaluator.py
──────────────────────────────
RAGAS evaluation harness for the RAG pipeline.
Two modes:
1. Offline: run against a test dataset (CI/CD regression gate)
2. Online: sample 1% of production queries asynchronously

BREAKING CHANGES in RAGAS 0.2.x vs 0.1.x:
  - `from ragas import evaluate` now accepts an EvaluationDataset, not a
    HuggingFace Dataset. Use `EvaluationDataset.from_list(...)`.
  - Metric imports moved: `from ragas.metrics import Faithfulness` (class, not instance).
  - evaluate() now uses `dataset=`, `metrics=[]` (list of metric instances), `llm=`, `embeddings=`.
  - LangchainLLMWrapper / LangchainEmbeddingsWrapper are in `ragas.llms` / `ragas.embeddings`.
  - result is an EvaluationResult; index with result["metric_name"] still works.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.storage.sql.postgres_store import get_postgres_store

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class RAGASEvaluator:
    """
    RAGAS 0.2.x-based evaluation.
    Uses GPT-4o-mini as the judge LLM (same as the RAG system).
    """

    def __init__(self):
        self._pg = get_postgres_store()

    def _get_ragas_components(self):
        """Lazy import RAGAS to avoid startup overhead."""
        # RAGAS 0.2.x imports
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

        # Instantiate metric objects (0.2.x uses classes, not module-level instances)
        base_metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]

        return (
            evaluate,
            EvaluationDataset,
            base_metrics,
            ContextRecall,  # pass class so caller can instantiate conditionally
            LangchainLLMWrapper(llm),
            LangchainEmbeddingsWrapper(embeddings),
        )

    async def evaluate_sample(self, sample: EvaluationSample) -> Dict[str, float]:
        """Evaluate a single query-answer pair."""
        (
            evaluate_fn,
            EvaluationDataset,
            base_metrics,
            ContextRecallClass,
            llm_wrapper,
            embed_wrapper,
        ) = self._get_ragas_components()

        # RAGAS 0.2.x expects a list of dicts with specific keys
        row = {
            "user_input": sample.question,          # key changed from "question" in 0.2.x
            "response": sample.answer,               # key changed from "answer" in 0.2.x
            "retrieved_contexts": sample.contexts,   # key changed from "contexts" in 0.2.x
        }
        metrics = list(base_metrics)

        if sample.ground_truth:
            row["reference"] = sample.ground_truth   # key changed from "ground_truth" in 0.2.x
            metrics.append(ContextRecallClass())

        # Build EvaluationDataset (replaces datasets.Dataset in 0.2.x)
        dataset = EvaluationDataset.from_list([row])

        try:
            result = evaluate_fn(
                dataset=dataset,
                metrics=metrics,
                llm=llm_wrapper,
                embeddings=embed_wrapper,
                raise_exceptions=False,
            )

            # result is an EvaluationResult; convert to DataFrame then extract row 0
            result_df = result.to_pandas()
            scores = {
                "faithfulness":      float(result_df.get("faithfulness", [0]).iloc[0]) if "faithfulness" in result_df else 0.0,
                "answer_relevancy":  float(result_df.get("answer_relevancy", [0]).iloc[0]) if "answer_relevancy" in result_df else 0.0,
                "context_precision": float(result_df.get("context_precision", [0]).iloc[0]) if "context_precision" in result_df else 0.0,
                "context_recall":    float(result_df.get("context_recall", [0]).iloc[0]) if "context_recall" in result_df else 0.0,
            }

            logger.info(
                "RAGAS evaluation complete",
                question=sample.question[:60],
                **{k: round(v, 3) for k, v in scores.items()},
            )

            # Persist to PostgreSQL
            await self._pg.log_evaluation({
                "query_text":        sample.question,
                "answer":            sample.answer,
                "faithfulness":      scores["faithfulness"],
                "answer_relevancy":  scores["answer_relevancy"],
                "context_precision": scores["context_precision"],
                "context_recall":    scores["context_recall"],
            })

            return scores

        except Exception as e:
            logger.error("RAGAS evaluation failed", error=str(e))
            return {}

    async def evaluate_batch(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, float]:
        """Evaluate a batch and return aggregate scores."""
        all_scores = []
        for sample in samples:
            scores = await self.evaluate_sample(sample)
            if scores:
                all_scores.append(scores)

        if not all_scores:
            return {}

        keys = all_scores[0].keys()
        return {
            k: round(sum(s[k] for s in all_scores) / len(all_scores), 4)
            for k in keys
        }

    async def maybe_evaluate_online(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> None:
        """
        Online evaluation: randomly sample RAGAS_SAMPLE_RATE % of queries.
        Runs asynchronously — doesn't block the response.
        """
        if not settings.ragas_enabled:
            return
        if random.random() > settings.ragas_sample_rate:
            return

        import asyncio
        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
        )
        asyncio.create_task(self.evaluate_sample(sample))


# ─────────────────────────────────────────────────────────────────────────────
# Offline Evaluation (CI/CD)
# ─────────────────────────────────────────────────────────────────────────────

async def run_offline_evaluation(
    test_file: str = "tests/evaluation/test_dataset.json",
    fail_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Load test dataset and run full RAGAS evaluation.
    Fails (raises) if any metric falls below fail_threshold.
    """
    import json
    from src.retrieval.pipeline import RetrievalPipeline

    with open(test_file) as f:
        test_data = json.load(f)

    pipeline = RetrievalPipeline()
    evaluator = RAGASEvaluator()
    samples = []

    for item in test_data:
        response = await pipeline.query(
            user_query=item["question"],
            user_roles=item.get("roles", ["EMPLOYEE"]),
        )
        sample = EvaluationSample(
            question=item["question"],
            answer=response.answer,
            contexts=[rc.chunk.text for rc in response.source_chunks],
            ground_truth=item.get("ground_truth"),
        )
        samples.append(sample)

    scores = await evaluator.evaluate_batch(samples)
    logger.info("Offline evaluation complete", scores=scores)

    for metric, score in scores.items():
        if score < fail_threshold:
            raise ValueError(
                f"Quality gate failed: {metric}={score:.3f} < threshold={fail_threshold}"
            )

    return scores
