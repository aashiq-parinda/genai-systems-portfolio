from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class FailureScenario(str, Enum):
    BASELINE = "baseline"
    BAD_RETRIEVAL = "bad_retrieval"
    HALLUCINATION = "hallucination"
    FAILED_TOOL = "failed_tool"

@dataclass
class ScenarioDefinition:
    scenario: FailureScenario
    title: str
    description: str
    user_query: str
    expected_failure_stage: str
    expected_root_cause: str

SCENARIOS: Dict[FailureScenario, ScenarioDefinition] = {
    FailureScenario.BASELINE: ScenarioDefinition(
        scenario=FailureScenario.BASELINE,
        title="Baseline: Successful RAG + Tool Pipeline",
        description="User asks for their order status and upgrade options. Retriever fetches billing doc, tool checks order ORD-88219, synthesis generates accurate grounded response.",
        user_query="Can you check my order ORD-88219 and tell me what the upgrade cost to Enterprise would be?",
        expected_failure_stage="none",
        expected_root_cause="None (Pipeline executes 100% healthy)"
    ),
    FailureScenario.BAD_RETRIEVAL: ScenarioDefinition(
        scenario=FailureScenario.BAD_RETRIEVAL,
        title="Failure Scenario 1: Bad / Irrelevant Retrieval",
        description="User asks about configuring SAML SSO on Enterprise plan. Retrieval returns marketing cookie policy. Context relevance drops drastically.",
        user_query="How do I configure SAML 2.0 Identity Provider with Okta and what is the ACS URL?",
        expected_failure_stage="retrieval",
        expected_root_cause="Vector store returned out-of-domain cookie tracking banner instead of sso_and_security.md"
    ),
    FailureScenario.HALLUCINATION: ScenarioDefinition(
        scenario=FailureScenario.HALLUCINATION,
        title="Failure Scenario 2: Hallucinated LLM Answer",
        description="Retriever fetches valid rate limit documentation. LLM is forced to fabricate non-existent 'Quantum-Encrypted Hyper-Clusters with free lifetime hardware upgrades'.",
        user_query="Does CloudFlow have hardware acceleration or special hosting options?",
        expected_failure_stage="generation",
        expected_root_cause="LLM fabricated claims unsupported by retrieved SaaS documentation; groundedness drops to < 0.20"
    ),
    FailureScenario.FAILED_TOOL: ScenarioDefinition(
        scenario=FailureScenario.FAILED_TOOL,
        title="Failure Scenario 3: Failed Tool Call (Downstream Timeout)",
        description="User inquires about order ORD-10492. Tool check_order_status encounters a simulated downstream 504 gateway timeout.",
        user_query="Please check the current payment status and seats for order ORD-10492.",
        expected_failure_stage="tool_execution",
        expected_root_cause="OrderServiceTimeoutError: Remote order database timed out after 5000ms"
    )
}
