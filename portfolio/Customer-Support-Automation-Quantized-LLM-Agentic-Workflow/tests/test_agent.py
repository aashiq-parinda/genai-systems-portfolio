"""
Unit tests for SupportAgentWorkflow Engine & State Execution DAG
"""
import pytest
from src.core.agent import SupportAgentWorkflow, WorkflowState
from src.core.quantization import QuantizationType


def test_agent_workflow_success():
    agent = SupportAgentWorkflow(quantization_type=QuantizationType.BITSANDBYTES_4BIT_NF4)
    res = agent.process_ticket("Can you check my shipment tracking status?")
    assert res["escalated"] is False
    assert res["state"] == WorkflowState.COMPLETED.value
    assert len(res["sources"]) > 0
    assert "Shipping" in res["sources"][0]
    assert res["latency_ms"] > 0
    assert res["prompt_tokens"] > 0
    assert res["completion_tokens"] > 0
    assert res["estimated_cost_usd"] > 0
    assert res["model_precision"] == "bnb_4bit_nf4"


def test_agent_workflow_escalation_security():
    agent = SupportAgentWorkflow()
    res = agent.process_ticket("There is unauthorized fraud on my account and wire theft.")
    assert res["escalated"] is True
    assert res["state"] == WorkflowState.ESCALATED.value
    assert "specialist" in res["response"].lower()
    assert res["sources"] == []


def test_agent_workflow_fp16_cost_delta():
    agent_4bit = SupportAgentWorkflow(quantization_type=QuantizationType.BITSANDBYTES_4BIT_NF4)
    agent_fp16 = SupportAgentWorkflow(quantization_type=QuantizationType.NONE)

    res_4bit = agent_4bit.process_ticket("What is the standard return policy?")
    res_fp16 = agent_fp16.process_ticket("What is the standard return policy?")

    assert res_fp16["estimated_cost_usd"] > res_4bit["estimated_cost_usd"]
    assert res_fp16["model_precision"] == "fp16"
