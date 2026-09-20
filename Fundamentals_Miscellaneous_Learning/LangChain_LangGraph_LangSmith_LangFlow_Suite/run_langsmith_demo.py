#!/usr/bin/env python3
"""
LangSmith Observability, Tracing & Evaluation Hands-On Demo.
Demonstrates:
1. Configuring LangSmith Telemetry Environment (LANGCHAIN_TRACING_V2)
2. Nesting Spans via @traceable Decorator
3. Visualizing Hierarchical Trace Waterfalls
4. Dataset Benchmark Evaluations with Faithfulness & SLA metrics
5. Syncing to Live smith.langchain.com Cloud Dashboard
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import Config
from src.langsmith_observability.tracing import (
    traceable,
    TraceContext,
    render_trace_waterfall
)
from src.langsmith_observability.evaluation import (
    LangSmithEvaluator,
    BENCHMARK_DATASET
)
from src.mock_llm import MockLLM

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

# Instrumented functions demonstrating nested parent-child trace runs
@traceable(name="vector_similarity_retriever", run_type="retriever", tags=["qdrant", "rag"])
def retrieve_context(query: str):
    time.sleep(0.04)
    return [
        "Qdrant HNSW parameters: m=16, ef_construct=100.",
        "BM25 sparse vectors provide keyword matching guarantees."
    ]

@traceable(name="llm_synthesis_call", run_type="llm", tags=["gpt-4o", "synthesis"])
def call_llm_synthesis(context: list, query: str):
    llm = MockLLM(model_name="gpt-4o", latency_ms=75)
    return llm.invoke(f"Context: {context}\nQuery: {query}").content

@traceable(name="output_sanitizer", run_type="chain", tags=["guardrail"])
def sanitize_output(text: str):
    time.sleep(0.01)
    return text.strip()

@traceable(name="enterprise_rag_pipeline", run_type="chain", tags=["production", "customer_support"])
def run_pipeline(query: str):
    ctx = retrieve_context(query)
    raw_synthesis = call_llm_synthesis(ctx, query)
    final_output = sanitize_output(raw_synthesis)
    return {"query": query, "answer": final_output, "intent": "general_knowledge"}

def demo_tracing():
    print_header("1. Capturing Nested Trace Run Tree via @traceable")
    Config.setup_langsmith_env()
    print(f"LangSmith Tracing Active: {Config.LANGCHAIN_TRACING_V2}")
    print(f"Target Project:           {Config.LANGCHAIN_PROJECT}")
    print(f"Endpoint:                 {Config.LANGCHAIN_ENDPOINT}")

    query = "How do we tune Qdrant HNSW parameters for 99.9% recall?"
    print(f"\nExecuting Pipeline for: \"{query}\"...")
    output = run_pipeline(query)

    print("\n" + "-" * 70)
    print("LANGSMITH SPAN WATERFALL (Hierarchical Execution Tree)")
    print("-" * 70)
    root_trace = TraceContext.get_latest_trace()
    if root_trace:
        print(render_trace_waterfall(root_trace))

def demo_evaluations():
    print_header("2. LangSmith Dataset Benchmark Evaluation Suite")
    print(f"Evaluating {len(BENCHMARK_DATASET)} test cases across 3 metrics:")
    print(" - Metric A: Intent Accuracy")
    print(" - Metric B: Groundedness / Faithfulness (0.0 - 1.0)")
    print(" - Metric C: Latency SLA Compliance (< 250ms)")

    evaluator = LangSmithEvaluator(target_pipeline=run_pipeline)
    eval_report = evaluator.run_benchmark(BENCHMARK_DATASET)

    print("\n--- INDIVIDUAL TEST CASE RUNS ---")
    for r in eval_report["results"]:
        status_icon = "✅" if r["overall_status"] == "PASSED" else "❌"
        print(
            f"{status_icon} [{r['test_id']}] {r['input_text']:<45} "
            f"Latency: {r['latency_ms']:>6.1f}ms | "
            f"Grounded: {r['groundedness_score']:.2f} | "
            f"Intent: {'MATCH' if r['intent_match'] else 'MISMATCH'}"
        )

    summary = eval_report["summary"]
    print("\n" + "=" * 70)
    print(" AGGREGATE EVALUATION SCORECARD")
    print("=" * 70)
    print(f"Total Test Cases:    {summary['total_runs']}")
    print(f"Passed Tests:        {summary['passed']} / {summary['total_runs']} ({summary['pass_rate']})")
    print(f"Average Latency:     {summary['avg_latency_ms']} ms")
    print(f"Mean Groundedness:   {summary['avg_groundedness']}")
    print(f"Mean Relevancy:      {summary['avg_relevance']}")

def demo_cloud_connection():
    print_header("3. Connecting to Official LangSmith Cloud Dashboard")
    print("To view live streaming traces in your browser on smith.langchain.com:")
    print("1. Sign up for free at https://smith.langchain.com")
    print("2. Generate an API key under Settings -> API Keys")
    print("3. Export your key in .env or terminal:")
    print("   export LANGCHAIN_TRACING_V2=true")
    print("   export LANGCHAIN_API_KEY=\"lsv2_pt_...\"")
    print("   export LANGCHAIN_PROJECT=\"lang-ecosystem-fundamentals\"")
    print("4. Any standard LangChain chain, LangGraph agent, or @traceable function")
    print("   will automatically stream telemetry directly to your cloud dashboard!")

if __name__ == "__main__":
    print("\n🚀 Starting LangSmith Observability & Evaluation Demonstrations...")
    demo_tracing()
    demo_evaluations()
    demo_cloud_connection()
    print("\n✅ LangSmith Demonstrations Completed Successfully!\n")
