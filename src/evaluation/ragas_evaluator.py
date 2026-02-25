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
    test_file: str = "src/evaluation/test_dataset.json",
    fail_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Load test dataset and run full RAGAS evaluation.

    SQL path and vector path are evaluated separately:
    - SQL queries have no source_chunks, so faithfulness/precision/recall
      are meaningless. Only answer_relevancy is scored for SQL.
    - Vector queries are scored on all 4 RAGAS metrics.

    Final scores are reported per-path and as a combined average
    (answer_relevancy only, since that applies to both paths).

    Fails (raises) if vector path metrics fall below fail_threshold.
    """
    import json
    from src.retrieval.pipeline import RetrievalPipeline

    with open(test_file) as f:
        test_data = json.load(f)

    pipeline  = RetrievalPipeline()
    evaluator = RAGASEvaluator()

    sql_samples    = []  # (item, sample) tuples
    vector_samples = []

    for item in test_data:
        query = item["query"]
        logger.info("Running pipeline query", query=query[:70])

        try:
            response = await pipeline.query(
                user_query=query,
                user_roles=item.get("roles", ["EMPLOYEE"]),
            )

            destination = (
                response.query_plan.route_decision.destination
                if response.query_plan and response.query_plan.route_decision
                else "VECTOR_STORE"
            )

            if destination == "SQL":
                # SQL path: no chunks returned.
                # Pass the answer as its own context — RAGAS can still score
                # answer_relevancy (does the answer address the question?).
                # faithfulness/precision/recall are skipped for SQL.
                sample = EvaluationSample(
                    question=query,
                    answer=response.answer,
                    contexts=[response.answer] if response.answer else ["No data returned."],
                    ground_truth=item.get("ground_truth"),
                )
                sql_samples.append(sample)

            else:
                # Vector path: use actual retrieved chunk texts as context.
                # Fall back to answer if somehow no chunks (e.g. "I don't have info").
                contexts = (
                    [rc.chunk.text for rc in response.source_chunks]
                    if response.source_chunks
                    else [response.answer]
                )
                sample = EvaluationSample(
                    question=query,
                    answer=response.answer,
                    contexts=contexts,
                    ground_truth=item.get("ground_truth"),
                )
                vector_samples.append(sample)

        except Exception as e:
            logger.error("Pipeline query failed", query=query[:60], error=str(e))

    # ── Evaluate SQL samples (answer_relevancy only) ──────────────────────
    sql_scores = {}
    if sql_samples:
        logger.info("Evaluating SQL path samples", count=len(sql_samples))
        sql_scores = await evaluator.evaluate_batch(sql_samples)
        logger.info(
            "SQL path evaluation complete",
            samples=len(sql_samples),
            answer_relevancy=sql_scores.get("answer_relevancy", 0),
        )

    # ── Evaluate vector samples (all 4 metrics) ───────────────────────────
    vector_scores = {}
    if vector_samples:
        logger.info("Evaluating vector path samples", count=len(vector_samples))
        vector_scores = await evaluator.evaluate_batch(vector_samples)
        logger.info(
            "Vector path evaluation complete",
            samples=len(vector_samples),
            **{k: round(v, 3) for k, v in vector_scores.items()},
        )

    # ── Build combined report ─────────────────────────────────────────────
    scores = {
        "sql_answer_relevancy":      round(sql_scores.get("answer_relevancy", 0),    4),
        "vector_faithfulness":       round(vector_scores.get("faithfulness", 0),      4),
        "vector_answer_relevancy":   round(vector_scores.get("answer_relevancy", 0),  4),
        "vector_context_precision":  round(vector_scores.get("context_precision", 0), 4),
        "vector_context_recall":     round(vector_scores.get("context_recall", 0),    4),
        "sql_sample_count":          len(sql_samples),
        "vector_sample_count":       len(vector_samples),
    }

    logger.info("Offline evaluation complete", scores=scores)

    # ── Print readable summary ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)
    print(f"\n📊 SQL Path ({len(sql_samples)} questions):")
    print(f"  answer_relevancy     : {scores['sql_answer_relevancy']:.4f}")
    print(f"\n📄 Vector Path ({len(vector_samples)} questions):")
    print(f"  faithfulness         : {scores['vector_faithfulness']:.4f}")
    print(f"  answer_relevancy     : {scores['vector_answer_relevancy']:.4f}")
    print(f"  context_precision    : {scores['vector_context_precision']:.4f}")
    print(f"  context_recall       : {scores['vector_context_recall']:.4f}")
    print("\n" + "=" * 55 + "\n")

    # ── Fail gate — only vector metrics (SQL has no retrieval to measure) ─
    vector_gate_metrics = {
        "vector_faithfulness":      scores["vector_faithfulness"],
        "vector_answer_relevancy":  scores["vector_answer_relevancy"],
        "vector_context_precision": scores["vector_context_precision"],
        "vector_context_recall":    scores["vector_context_recall"],
    }
    failures = {k: v for k, v in vector_gate_metrics.items() if v < fail_threshold}
    if failures:
        raise ValueError(
            f"Quality gate failed — metrics below threshold ({fail_threshold}):\n"
            + "\n".join(f"  {k}: {v:.4f}" for k, v in failures.items())
        )

    return scores

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS offline evaluation")
    parser.add_argument(
        "--dataset",
        default="src/evaluation/test_dataset.json",
        help="Path to evaluation dataset JSON",
    )   
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Fail if any metric falls below this score",
    )
    args = parser.parse_args()

    scores = asyncio.run(run_offline_evaluation(
        test_file=args.dataset,
        fail_threshold=args.threshold,
    ))
    print(scores)
