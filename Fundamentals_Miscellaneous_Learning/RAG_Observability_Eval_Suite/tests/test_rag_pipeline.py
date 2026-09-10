import pytest
from pathlib import Path
from src.config import settings
from src.vector_store import VectorStore
from src.tools import check_order_status, OrderServiceTimeoutError, OrderLookupError
from src.evaluation.llm_judge import llm_judge
from src.failure_injection import FailureScenario
from src.rag_engine import RAGPipeline
from src.telemetry.langfuse_client import langfuse_client
from src.telemetry.phoenix_client import phoenix_client
from src.telemetry.helicone_client import helicone_client
from src.telemetry.otel_tracer import (
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_RETRIEVAL_QUERY,
    GEN_AI_TOOL_NAME
)

@pytest.fixture
def rag_pipeline():
    return RAGPipeline()

def test_vector_store_indexing_and_retrieval():
    vs = VectorStore(docs_dir=settings.docs_dir)
    assert len(vs.chunks) > 0
    
    # Query for billing/plans
    chunks = vs.retrieve("What is the cost of Enterprise subscription tier?", top_k=2)
    assert len(chunks) == 2
    assert "billing_and_plans.md" in [c.source_file for c in chunks]
    assert chunks[0].score > 0.0

def test_vector_store_bad_retrieval_injection():
    vs = VectorStore(docs_dir=settings.docs_dir)
    chunks = vs.retrieve("How to configure Okta SSO?", top_k=2, force_irrelevant=True)
    assert len(chunks) == 1
    assert "cookie" in chunks[0].content.lower()

def test_mock_tool_order_lookup_success():
    order = check_order_status("ORD-88219")
    assert order["status"] == "active"
    assert order["plan"] == "Enterprise"
    assert order["amount_usd"] == 1200.00

def test_mock_tool_order_lookup_failure_injection():
    with pytest.raises(OrderServiceTimeoutError):
        check_order_status("ORD-88219", force_failure=True, failure_type="timeout")

def test_llm_judge_groundedness_evaluation():
    # Grounded answer
    good_ctx = "CloudFlow Enterprise offers SAML 2.0 and Okta integration with 99.95% uptime."
    good_ans = "CloudFlow Enterprise supports Okta SAML 2.0 with a 99.95% uptime guarantee."
    res = llm_judge.evaluate_groundedness(good_ans, good_ctx)
    assert res.passed is True
    assert res.score >= 0.60

    # Hallucinated answer
    hallucinated_ans = "CloudFlow provides Quantum-Encrypted Hyper-Clusters with free lifetime hardware upgrades."
    bad_res = llm_judge.evaluate_groundedness(hallucinated_ans, good_ctx)
    assert bad_res.passed is False
    assert bad_res.score < 0.30

def test_llm_judge_context_relevance_evaluation():
    query = "How to configure Okta SAML 2.0?"
    good_ctx = "CloudFlow supports Okta SAML 2.0 integration under Settings > Security & SSO."
    res = llm_judge.evaluate_context_relevance(query, good_ctx)
    assert res.passed is True

    bad_ctx = "We use third-party tracking cookies to analyze marketing bounce rates."
    bad_res = llm_judge.evaluate_context_relevance(query, bad_ctx)
    assert bad_res.passed is False
    assert bad_res.score <= 0.20

def test_pipeline_baseline_run(rag_pipeline):
    result = rag_pipeline.execute_query(
        query="Can you check order ORD-88219 and tell me what the upgrade cost to Enterprise would be?",
        scenario=FailureScenario.BASELINE
    )
    assert result["scenario"] == "baseline"
    assert result["tool_error"] is None
    assert result["tool_result"]["order_id"] == "ORD-88219"
    assert result["evaluations"]["groundedness"]["passed"] is True
    assert result["evaluations"]["context_relevance"]["passed"] is True
    assert result["spans_count"] >= 3

def test_pipeline_failure_scenario_1_bad_retrieval(rag_pipeline):
    result = rag_pipeline.execute_query(
        query="How do I configure SAML 2.0 Identity Provider with Okta?",
        scenario=FailureScenario.BAD_RETRIEVAL
    )
    assert result["scenario"] == "bad_retrieval"
    assert result["evaluations"]["context_relevance"]["passed"] is False
    assert result["evaluations"]["context_relevance"]["score"] <= 0.20

def test_pipeline_failure_scenario_2_hallucination(rag_pipeline):
    result = rag_pipeline.execute_query(
        query="Does CloudFlow have hardware acceleration or special hosting options?",
        scenario=FailureScenario.HALLUCINATION
    )
    assert result["scenario"] == "hallucination"
    assert result["evaluations"]["groundedness"]["passed"] is False
    assert result["evaluations"]["groundedness"]["score"] <= 0.20

def test_pipeline_failure_scenario_3_failed_tool(rag_pipeline):
    result = rag_pipeline.execute_query(
        query="Please check order ORD-10492 status.",
        scenario=FailureScenario.FAILED_TOOL
    )
    assert result["scenario"] == "failed_tool"
    assert result["tool_error"] is not None
    assert "timeout" in result["tool_error"].lower()

def test_multi_sink_observability_sync(rag_pipeline):
    # Execute a run
    res = rag_pipeline.execute_query(
        query="What are the rate limits for the Starter plan?",
        scenario=FailureScenario.BASELINE
    )
    trace_id = res["trace_id"]
    
    # 1. Check Langfuse trace and scores
    lf_trace = langfuse_client.get_trace_summary(trace_id)
    assert lf_trace is not None
    assert len(lf_trace["spans"]) >= 2
    assert any(s["name"] == "groundedness" for s in lf_trace["scores"])

    # 2. Check Arize Phoenix trace
    ph_trace = phoenix_client.get_trace_summary(trace_id)
    assert ph_trace is not None
    assert ph_trace.retrieval_metrics["chunks_retrieved_count"] > 0
    assert "groundedness" in ph_trace.evaluations

    # 3. Check Helicone proxy logs
    last_req = helicone_client.request_logs[-1]
    assert last_req.request_id == trace_id
    assert last_req.cost_usd >= 0.0
    assert "Helicone-Auth" in last_req.headers
