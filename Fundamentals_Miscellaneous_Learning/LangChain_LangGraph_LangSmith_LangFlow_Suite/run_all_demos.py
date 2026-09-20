#!/usr/bin/env python3
"""
Master Verification Runner for LangChain, LangGraph, LangSmith, and LangFlow Suite.
Executes end-to-end validation across all 4 pillars and reports overall system health.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import Config
from src.langchain_core.lcel_pipeline import build_lcel_intent_classifier, build_lcel_parallel_analyzer
from src.langgraph_agents.workflow import build_vector_db_research_graph
from src.langgraph_agents.checkpointing import Checkpointer
from src.langsmith_observability.tracing import TraceContext, traceable
from src.langsmith_observability.evaluation import LangSmithEvaluator, BENCHMARK_DATASET
from src.langflow_flows.flow_exporter import validate_flow_json

def run_suite():
    print("=" * 75)
    print(" 🚀 EXECUTING MASTER SUITE VERIFICATION: LANGCHAIN, LANGGRAPH, LANGSMITH, LANGFLOW")
    print("=" * 75)
    results = {}

    # Test 1: LangChain LCEL
    try:
        t0 = time.time()
        chain = build_lcel_intent_classifier()
        res = chain.invoke({"text": "How much does a 10M vector cluster cost?"})
        assert res.intent == "billing_inquiry", f"Unexpected intent: {res.intent}"
        parallel = build_lcel_parallel_analyzer()
        p_res = parallel.invoke({"text": "Urgent performance audit"})
        assert "summary" in p_res and "key_entities" in p_res
        results["LangChain LCEL & Parallel"] = {"status": "PASSED", "duration_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        results["LangChain LCEL & Parallel"] = {"status": f"FAILED: {e}", "duration_ms": 0}

    # Test 2: LangGraph Cyclic State Machine & Checkpointer
    try:
        t0 = time.time()
        workflow = build_vector_db_research_graph()
        checkpointer = Checkpointer()
        thread_id = "test_thread_verify"
        state = {
            "query": "Test LangGraph Cyclic Workflow",
            "thread_id": thread_id,
            "plan": [],
            "research_notes": [],
            "messages": [],
            "draft": "",
            "critique": "",
            "quality_score": 0.0,
            "revision_count": 0,
            "human_approved": False,
            "status": "init"
        }
        current = workflow.entry_point
        steps_executed = 0
        while current and steps_executed < 15:
            if current in workflow.interrupt_before and not state.get("human_approved", False):
                # Simulate human intervention
                state["human_approved"] = True
            node_name, updates, state, next_node = workflow.stream_step(current, state)
            checkpointer.save(thread_id, node_name, state)
            current = next_node
            steps_executed += 1

        assert state["status"] == "approved_and_completed"
        assert state["revision_count"] >= 2
        results["LangGraph Cyclic Agent & Checkpointer"] = {"status": "PASSED", "duration_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        results["LangGraph Cyclic Agent & Checkpointer"] = {"status": f"FAILED: {e}", "duration_ms": 0}

    # Test 3: LangSmith Tracing & Dataset Evaluation
    try:
        t0 = time.time()
        @traceable(name="test_traceable_function", run_type="chain")
        def test_fn(x: str):
            return {"intent": "general_knowledge", "text": x}

        evaluator = LangSmithEvaluator(target_pipeline=test_fn)
        eval_res = evaluator.run_benchmark(BENCHMARK_DATASET)
        assert eval_res["summary"]["total_runs"] == len(BENCHMARK_DATASET)
        assert TraceContext.get_latest_trace() is not None
        results["LangSmith Tracing & Evaluator"] = {"status": "PASSED", "duration_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        results["LangSmith Tracing & Evaluator"] = {"status": f"FAILED: {e}", "duration_ms": 0}

    # Test 4: LangFlow Flow Topologies
    try:
        t0 = time.time()
        flow1 = validate_flow_json(BASE_DIR / "flows" / "01_lcel_conversational_rag.json")
        flow2 = validate_flow_json(BASE_DIR / "flows" / "02_langgraph_cyclic_agent.json")
        assert flow1["is_valid"] and flow2["is_valid"], "Flow schema validation failed"
        results["LangFlow Visual Flows Validation"] = {"status": "PASSED", "duration_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        results["LangFlow Visual Flows Validation"] = {"status": f"FAILED: {e}", "duration_ms": 0}

    # Print Summary Scorecard
    print("\nMASTER VERIFICATION SUMMARY:")
    all_passed = True
    for module, data in results.items():
        icon = "✅" if data["status"] == "PASSED" else "❌"
        if data["status"] != "PASSED":
            all_passed = False
        print(f" {icon} {module:<42} -> {data['status']:<15} ({data['duration_ms']} ms)")

    print("=" * 75)
    if all_passed:
        print("🎉 ALL 4 CORE PILLARS VERIFIED AND READY FOR EXPERIMENTATION!")
    else:
        print("⚠️ Some modules encountered errors during verification.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_suite()
