"""
LangSmith Dataset Evaluation & Benchmark Suite.
Demonstrates:
1. Dataset Management (Test Cases with Inputs & Expected Criteria)
2. Custom Evaluators (Faithfulness, Relevancy, Latency SLA Compliance)
3. Running Benchmark Evaluations and Generating Summary Scorecards
"""

import time
from typing import Dict, Any, List, Callable
from pydantic import BaseModel
from ..mock_llm import MockLLM

class TestCase(BaseModel):
    id: str
    input_text: str
    expected_intent: str
    must_contain: List[str]
    max_latency_ms: float

class EvalResult(BaseModel):
    test_id: str
    input_text: str
    intent_match: bool
    groundedness_score: float
    relevance_score: float
    latency_ms: float
    passed_sla: bool
    overall_status: str

# Sample benchmark dataset
BENCHMARK_DATASET: List[TestCase] = [
    TestCase(
        id="TC-01",
        input_text="How much does an enterprise vector database cluster cost per month?",
        expected_intent="billing_inquiry",
        must_contain=["cost", "pricing"],
        max_latency_ms=250.0
    ),
    TestCase(
        id="TC-02",
        input_text="Our HNSW indexing job crashed with out-of-memory error 137.",
        expected_intent="technical_support",
        must_contain=["memory", "crash"],
        max_latency_ms=250.0
    ),
    TestCase(
        id="TC-03",
        input_text="What are the main differences between Pinecone, Milvus, and Qdrant?",
        expected_intent="general_knowledge",
        must_contain=["Qdrant", "Pinecone", "Milvus"],
        max_latency_ms=300.0
    ),
    TestCase(
        id="TC-04",
        input_text="Can we configure hybrid search with BM25 sparse vectors and dense vectors?",
        expected_intent="general_knowledge",
        must_contain=["hybrid", "BM25"],
        max_latency_ms=250.0
    )
]

class LangSmithEvaluator:
    """
    Evaluator harness mirroring LangSmith's evaluate() function.
    Runs pipeline against a dataset and computes aggregate performance metrics.
    """
    def __init__(self, target_pipeline: Callable[[str], Dict[str, Any]]):
        self.target_pipeline = target_pipeline

    def run_benchmark(self, dataset: List[TestCase] = BENCHMARK_DATASET) -> Dict[str, Any]:
        results: List[EvalResult] = []

        for tc in dataset:
            t0 = time.time()
            output = self.target_pipeline(tc.input_text)
            latency_ms = (time.time() - t0) * 1000.0

            # 1. Intent Match Evaluator
            predicted_intent = output.get("intent", "")
            intent_match = (predicted_intent.lower() == tc.expected_intent.lower())

            # 2. Groundedness / Relevancy Evaluator
            groundedness_score = 0.95 if any(kw.lower() in str(output).lower() for kw in tc.must_contain) else 0.82
            relevance_score = 0.94 if intent_match else 0.65

            # 3. Latency SLA Evaluator
            passed_sla = latency_ms <= tc.max_latency_ms

            passed_all = intent_match and groundedness_score >= 0.85 and passed_sla
            overall_status = "PASSED" if passed_all else "FAILED"

            results.append(EvalResult(
                test_id=tc.id,
                input_text=tc.input_text[:40] + "...",
                intent_match=intent_match,
                groundedness_score=groundedness_score,
                relevance_score=relevance_score,
                latency_ms=round(latency_ms, 2),
                passed_sla=passed_sla,
                overall_status=overall_status
            ))

        # Compute aggregates
        total = len(results)
        passed_count = sum(1 for r in results if r.overall_status == "PASSED")
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0
        avg_groundedness = sum(r.groundedness_score for r in results) / total if total else 0.0
        avg_relevance = sum(r.relevance_score for r in results) / total if total else 0.0

        return {
            "summary": {
                "total_runs": total,
                "passed": passed_count,
                "pass_rate": f"{(passed_count / total * 100):.1f}%" if total else "0%",
                "avg_latency_ms": round(avg_latency, 2),
                "avg_groundedness": round(avg_groundedness, 3),
                "avg_relevance": round(avg_relevance, 3)
            },
            "results": [r.model_dump() for r in results]
        }
