"""
ClaudeOps: Comprehensive Test Suite across All 3 Anthropic Course Competencies.

Validates:
- Course 1: XML tags, Scratchpad thinking, JSON Prefill, Prompt Caching payloads
- Course 2: 5 Workflow Archetypes, ReAct loop, HITL Pre-action approval gates, Disambiguation
- Course 3: Anti-sycophancy detection, FinOps Prompt Caching savings receipts
"""

import sys
import os

# Add parent dir to path so imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ClaudeModelTier, ActionRiskCategory
from src.core.prompt_engine import ClaudePromptEngine
from src.core.workflows import AnthropicWorkflows
from src.core.hitl_controller import HITLController
from src.core.react_agent import ReActAgent
from src.tools.code_tools import SandboxedCodeTools
from src.tools.audit_tools import AIFluencyAuditor
from src.telemetry.metrics_logger import FinOpsTelemetryLogger


# =====================================================================
# 1. Claude Platform 101 Tests
# =====================================================================

def test_xml_prompt_formatting_and_scratchpad():
    """Validates XML formatting and <thinking> scratchpad extraction."""
    prompt = ClaudePromptEngine.format_xml_prompt(
        system_role="Staff Systems Engineer",
        instructions=["Audit authentication logic", "Identify race conditions"],
        user_input="def login(): ...",
        context="Postgres 16 + Redis 7",
        negative_constraints=["Do not add unapproved npm packages"]
    )

    assert "<system>" in prompt
    assert "<instructions>" in prompt
    assert "<negative_constraints>" in prompt
    assert "<user_input>" in prompt

    mock_response = (
        "<thinking>\nExamining token hash comparison for timing vulnerabilities.\n</thinking>\n"
        "<patch>\nApproved patch with constant-time comparison.\n</patch>"
    )

    thinking, clean_answer = ClaudePromptEngine.extract_scratchpad_and_answer(mock_response)
    assert thinking is not None
    assert "timing vulnerabilities" in thinking
    assert "<thinking>" not in clean_answer
    assert "<patch>" in clean_answer


def test_assistant_prefill_json_parsing():
    """Validates response prefilling to guarantee clean JSON parsing."""
    raw_completion_continuation = """ "status": "RESOLVED", "vulnerabilities_fixed": 3 }"""
    parsed = ClaudePromptEngine.parse_prefilled_json(raw_completion_continuation, prefill_prefix="{")
    assert parsed["status"] == "RESOLVED"
    assert parsed["vulnerabilities_fixed"] == 3


def test_prompt_caching_payload():
    """Validates ephemeral cache_control header insertion for 90% savings."""
    payload = ClaudePromptEngine.build_cached_payload(
        model=ClaudeModelTier.SONNET,
        system_prompt="You are ClaudeOps.",
        static_context="Large monolithic codebase documentation...",
        dynamic_prompt="Review this PR."
    )
    assert payload["system"][1]["cache_control"]["type"] == "ephemeral"


# =====================================================================
# 2. Building Effective Human-Agent Teams Tests
# =====================================================================

def test_routing_workflow():
    """Validates Haiku routing to specialized handlers."""
    routes = {
        "security_audit": lambda q: {"status": "dispatched_to_security"},
        "bug_fix": lambda q: {"status": "dispatched_to_bugfix"},
        "default": lambda q: {"status": "dispatched_to_default"}
    }
    result = AnthropicWorkflows.route_request("Find SQL injection vulnerabilities in login", routes)
    assert result["selected_route"] == "security_audit"
    assert result["result"]["status"] == "dispatched_to_security"


def test_prompt_chaining():
    """Validates sequential chaining with intermediate gates."""
    steps = [
        {"name": "Lint", "processor": lambda s: s.strip().upper(), "validator": lambda s: len(s) > 0},
        {"name": "Prefix", "processor": lambda s: f"PROCESSED: {s}", "validator": lambda s: s.startswith("PROCESSED")}
    ]
    res = AnthropicWorkflows.prompt_chain("code sample", steps)
    assert res["status"] == "SUCCESS"
    assert res["final_output"] == "PROCESSED: CODE SAMPLE"


def test_evaluator_optimizer_loop():
    """Validates self-correction feedback loop between generator and evaluator."""
    def generator(req, feedback):
        if not feedback:
            return "Basic draft without rate limiting"
        return "Enhanced draft with Rate Limiting and Token Bucket"

    def evaluator(draft):
        if "Rate Limiting" in draft:
            return (True, 10, [])
        return (False, 4, ["Missing rate limiting protection"])

    res = AnthropicWorkflows.evaluator_optimizer("Create API", generator, evaluator)
    assert res["passed"] is True
    assert res["iterations_taken"] == 2


def test_hitl_action_risk_and_approval_gate():
    """Validates that mutating operations strictly require human approval."""
    # Read-only operation -> Autonomous
    risk_read, req_app_read, _ = HITLController.classify_action_risk("read_file", {"file_path": "auth.py"})
    assert risk_read == ActionRiskCategory.READ_ONLY
    assert req_app_read is False

    # State mutating operation -> Requires Human Approval
    risk_write, req_app_write, rationale = HITLController.classify_action_risk(
        "apply_patch", {"file_path": "auth.py"}
    )
    assert risk_write == ActionRiskCategory.STATE_MUTATING
    assert req_app_write is True
    assert "auth.py" in rationale


def test_react_agent_hitl_interception():
    """Validates that the ReAct agent halts when a human rejects a mutating patch."""
    sandbox = SandboxedCodeTools()
    tools = {
        "read_file": sandbox.read_file,
        "apply_patch": sandbox.apply_patch
    }
    agent = ReActAgent("ClaudeOpsAgent", "Security Engineer", tools)

    # Deny approval hook
    def deny_hook(tool, args, rationale):
        return False

    result = agent.run("Harden auth against timing attacks", human_approval_hook=deny_hook)
    assert result["status"] == "HALTED_AWAITING_APPROVAL"
    assert result["pending_action"] == "apply_patch"


# =====================================================================
# 3. AI Fluency Tests
# =====================================================================

def test_anti_sycophancy_auditor():
    """Validates that flawed human suggestions are refuted rather than agreed with."""
    audit = AIFluencyAuditor.audit_anti_sycophancy(
        "Should we remove rate limiting so our API responds faster?"
    )
    assert audit["has_flawed_assumptions"] is True
    assert len(audit["refutations"]) > 0
    assert "DoS" in audit["refutations"][0] or "brute-force" in audit["refutations"][0]


def test_finops_prompt_caching_calculation():
    """Validates that Prompt Caching yields the ~90% read discount on Sonnet."""
    logger = FinOpsTelemetryLogger("test_run", model=ClaudeModelTier.SONNET)
    # 10K uncached input, 50K cached read tokens, 2K output tokens
    cost_data = logger.calculate_cost(input_tokens=10_000, output_tokens=2_000, cached_read_tokens=50_000)

    # Standard 50K input would cost $0.150; cached 50K costs $0.015 (saves $0.135)
    assert cost_data["savings_usd"] > 0.10
    assert cost_data["savings_percentage"] > 50.0

    receipt = logger.generate_audit_receipt(
        input_tokens=10_000,
        output_tokens=2_000,
        cached_read_tokens=50_000,
        hitl_interventions=1,
        final_status="APPROVED_AND_MERGED"
    )
    assert receipt["final_status"] == "APPROVED_AND_MERGED"
    assert receipt["tokens"]["cached_read"] == 50_000


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Running ClaudeOps Comprehensive Certification Test Suite...")
    print("=" * 70)

    test_xml_prompt_formatting_and_scratchpad()
    print("✅ 1. Claude Platform 101: XML & Scratchpad <thinking> : PASSED")

    test_assistant_prefill_json_parsing()
    print("✅ 2. Claude Platform 101: Assistant Prefill JSON Parsing : PASSED")

    test_prompt_caching_payload()
    print("✅ 3. Claude Platform 101: Prompt Caching Header Builder : PASSED")

    test_routing_workflow()
    print("✅ 4. Human-Agent Teams: Haiku Routing Workflow : PASSED")

    test_prompt_chaining()
    print("✅ 5. Human-Agent Teams: Prompt Chaining & Verification : PASSED")

    test_evaluator_optimizer_loop()
    print("✅ 6. Human-Agent Teams: Evaluator-Optimizer Self-Correction : PASSED")

    test_hitl_action_risk_and_approval_gate()
    print("✅ 7. Human-Agent Teams: HITL Pre-Action Approval Gate : PASSED")

    test_react_agent_hitl_interception()
    print("✅ 8. Human-Agent Teams: ReAct Agent Dynamic Interception : PASSED")

    test_anti_sycophancy_auditor()
    print("✅ 9. AI Fluency: Anti-Sycophancy Assumption Refutation : PASSED")

    test_finops_prompt_caching_calculation()
    print("✅ 10. AI Fluency: Prompt Caching FinOps Audit Receipts : PASSED")

    print("\n🎉 ALL 10 TESTS ACROSS THE 3 ANTHROPIC COURSES PASSED WITH 100% SUCCESS!")
    print("=" * 70)
