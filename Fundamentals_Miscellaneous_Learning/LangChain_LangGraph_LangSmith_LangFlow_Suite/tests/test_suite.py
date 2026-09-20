"""
Pytest Test Suite for LangChain, LangGraph, LangSmith, and LangFlow Suite.
"""

import sys
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.langchain_core.lcel_pipeline import (
    build_lcel_intent_classifier,
    build_lcel_parallel_analyzer,
    build_lcel_resilient_fallback_chain
)
from src.langchain_core.memory_and_history import (
    ConversationBuffer,
    trim_messages
)
from src.langgraph_agents.workflow import build_vector_db_research_graph
from src.langgraph_agents.checkpointing import Checkpointer
from src.langsmith_observability.tracing import TraceContext, traceable
from src.langsmith_observability.evaluation import LangSmithEvaluator, BENCHMARK_DATASET
from src.langflow_flows.flow_exporter import validate_flow_json

FLOWS_DIR = Path(__file__).resolve().parent.parent / "flows"

def test_lcel_pipeline():
    chain = build_lcel_intent_classifier()
    res = chain.invoke({"text": "How much does a 10M vector cluster cost?"})
    assert res.intent == "billing_inquiry"
    assert res.confidence > 0.8

def test_lcel_parallel_and_fallbacks():
    parallel = build_lcel_parallel_analyzer()
    p_res = parallel.invoke({"text": "Urgent performance audit for vector database"})
    assert "summary" in p_res
    assert "key_entities" in p_res
    assert p_res["estimated_priority"] == "P1-High"

    fallback_chain = build_lcel_resilient_fallback_chain()
    fb_res = fallback_chain.invoke({"text": "Test failover"})
    assert "Fallback" in fb_res

def test_memory_trimming():
    buf = ConversationBuffer()
    buf.add_system_message("System prompt")
    buf.add_user_message("Query 1")
    buf.add_ai_message("Answer 1")
    buf.add_user_message("Query 2")
    buf.add_ai_message("Answer 2")
    
    trimmed = trim_messages(buf.get_messages(), max_messages=2, keep_system=True)
    assert len(trimmed) == 3
    assert trimmed[0].role == "system"

def test_langgraph_cyclic_workflow():
    workflow = build_vector_db_research_graph()
    checkpointer = Checkpointer()
    thread_id = "test_thread_pytest"

    state = {
        "query": "Compare Pinecone and Qdrant",
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
    steps = 0
    while current and steps < 12:
        if current in workflow.interrupt_before and not state.get("human_approved", False):
            # Verify interrupt triggered
            assert current == "approval_gate"
            state["human_approved"] = True

        node_name, updates, state, next_node = workflow.stream_step(current, state)
        checkpointer.save(thread_id, node_name, state)
        current = next_node
        steps += 1

    assert state["status"] == "approved_and_completed"
    assert state["revision_count"] >= 2
    assert len(checkpointer.get_history(thread_id)) >= 5

def test_langsmith_tracing_and_eval():
    @traceable(name="pytest_trace_root", run_type="chain")
    def sample_pipeline(text: str):
        return {"intent": "general_knowledge", "text": text}

    evaluator = LangSmithEvaluator(target_pipeline=sample_pipeline)
    summary = evaluator.run_benchmark(BENCHMARK_DATASET)
    assert summary["summary"]["total_runs"] == len(BENCHMARK_DATASET)

def test_langflow_flow_schemas():
    flow1 = validate_flow_json(FLOWS_DIR / "01_lcel_conversational_rag.json")
    flow2 = validate_flow_json(FLOWS_DIR / "02_langgraph_cyclic_agent.json")
    assert flow1["is_valid"] is True
    assert flow2["is_valid"] is True
    assert flow1["node_count"] == 5
    assert flow2["node_count"] == 5
