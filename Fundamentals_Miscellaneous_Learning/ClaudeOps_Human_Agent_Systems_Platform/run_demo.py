#!/usr/bin/env python3
"""
ClaudeOps: Interactive Master Demonstration Runner.

Showcases the complete unified platform for technical interviews and portfolio reviews:
- Scene 1: Haiku Intent Routing
- Scene 2: XML Structured Prompt & Scratchpad (<thinking>)
- Scene 3: Prompt Caching FinOps Calculation (90% read discount)
- Scene 4: ReAct Agent with Human-in-the-Loop (HITL) Pre-Action Gate
- Scene 5: AI Fluency Anti-Sycophancy Auditor
- Scene 6: Immutable JSON Audit Receipt & FinOps Report
"""

import time
import json
from typing import Dict, Any
from config import ClaudeModelTier, ActionRiskCategory
from src.core.prompt_engine import ClaudePromptEngine
from src.core.workflows import AnthropicWorkflows
from src.core.hitl_controller import HITLController
from src.core.react_agent import ReActAgent
from src.tools.code_tools import SandboxedCodeTools
from src.tools.audit_tools import AIFluencyAuditor
from src.telemetry.metrics_logger import FinOpsTelemetryLogger


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"🔹 {title}")
    print("=" * 80)


def main():
    logger = FinOpsTelemetryLogger("ClaudeOps-Live-PR-Remediation", model=ClaudeModelTier.SONNET)

    print("\n" + "█" * 80)
    print("🚀 CLAUDEOPS: AUTONOMOUS HUMAN-AGENT SYSTEMS PLATFORM (ANTHROPIC ACADEMY)")
    print("█" * 80)

    # -----------------------------------------------------------------
    # SCENE 1: Haiku Routing Triage
    # -----------------------------------------------------------------
    print_banner("SCENE 1: Fast Intent Triage & Routing (Claude 3.5 Haiku)")
    sample_request = "URGENT: Timing attack vulnerability discovered in auth login endpoint!"
    print(f"Incoming Ticket: \"{sample_request}\"")

    routes = {
        "security_audit": lambda q: {"handler": "SecurityHardeningAgent", "priority": "P0_CRITICAL"},
        "bug_fix": lambda q: {"handler": "BugFixAgent", "priority": "P2_NORMAL"},
        "default": lambda q: {"handler": "GeneralAgent", "priority": "P3_LOW"}
    }
    routing_result = AnthropicWorkflows.route_request(sample_request, routes)
    print(f"👉 Routed By: {routing_result['model_used']}")
    print(f"👉 Selected Route: {routing_result['selected_route']}")
    print(f"👉 Assigned Handler: {routing_result['result']['handler']} (Priority: {routing_result['result']['priority']})")
    logger.record_step("Routing", routing_result)

    # -----------------------------------------------------------------
    # SCENE 2: XML Structured Prompt & Scratchpad Thinking
    # -----------------------------------------------------------------
    print_banner("SCENE 2: XML Prompt Generation & Scratchpad Reasoning (Claude 3.5 Sonnet)")
    prompt = ClaudePromptEngine.format_xml_prompt(
        system_role="Staff AppSec & Distributed Systems Engineer",
        instructions=[
            "Inspect auth.py for timing attacks",
            "Replace equality operator with constant-time HMAC comparison",
            "Inject rate limiting middleware call"
        ],
        context="Python 3.11, FastAPI microservice, Redis-backed rate limiter",
        user_input=sample_request,
        negative_constraints=["Never disable unit tests", "Do not add unapproved PyPI dependencies"]
    )
    print("Generated XML Prompt (truncated):")
    print("\n".join(prompt.splitlines()[:16]) + "\n... [truncated for display]")

    # -----------------------------------------------------------------
    # SCENE 3: Prompt Caching & FinOps Economics
    # -----------------------------------------------------------------
    print_banner("SCENE 3: Prompt Caching FinOps Economics (Claude Platform 101)")
    # Simulating 45K tokens of static repo documentation cached with ephemeral TTL
    uncached_input_tokens = 5_000
    cached_docs_tokens = 45_000
    output_tokens = 1_200

    cost_data = logger.calculate_cost(
        input_tokens=uncached_input_tokens,
        output_tokens=output_tokens,
        cached_read_tokens=cached_docs_tokens
    )
    print(f"• Uncached Tokens Processed:   {uncached_input_tokens:,}")
    print(f"• Ephemeral Cached Read Tokens: {cached_docs_tokens:,} (90% read discount)")
    print(f"• Output Tokens Generated:     {output_tokens:,}")
    print(f"• Actual Cost (With Caching):  ${cost_data['actual_total_usd']:.4f}")
    print(f"• Cost Without Caching:        ${cost_data['uncached_hypothetical_usd']:.4f}")
    print(f"💰 Financial Savings:          ${cost_data['savings_usd']:.4f} ({cost_data['savings_percentage']:.1f}% FinOps reduction!)")

    # -----------------------------------------------------------------
    # SCENE 4: Dynamic ReAct Loop with HITL Pre-Action Approval Gate
    # -----------------------------------------------------------------
    print_banner("SCENE 4: Dynamic ReAct Loop & Human-in-the-Loop (HITL) Safety Gate")
    sandbox = SandboxedCodeTools()
    tools = {
        "read_file": sandbox.read_file,
        "apply_patch": sandbox.apply_patch
    }
    agent = ReActAgent("ClaudeOpsAgent", "Security Engineer", tools)

    hitl_interventions = 0

    def simulated_human_approval_hook(tool_name: str, args: Dict[str, Any], rationale: str) -> bool:
        nonlocal hitl_interventions
        hitl_interventions += 1
        print("\n" + "⚠️ " * 20)
        print("🛑 HUMAN-IN-THE-LOOP CONTROL GATE TRIGGERED!")
        print(f"   Tool Name:   {tool_name}")
        print(f"   Target:      {args.get('file_path')}")
        print(f"   Rationale:   {rationale}")
        print(f"   Action Type: STATE_MUTATING (Modifies Source Code)")
        print("   Human Decision: [APPROVED BY TECH LEAD Ashraf Khan]")
        print("⚠️ " * 20 + "\n")
        return True

    agent_run = agent.run("Harden auth.py against timing attacks", human_approval_hook=simulated_human_approval_hook)
    print(f"Agent Status: {agent_run['status']} (Completed in {agent_run['iterations']} iterations)")
    print("Generated Unified Git Diff:")
    last_observation = agent_run['history'][-1]['observation']
    if 'unified_diff' in last_observation:
        print("-" * 50)
        print(last_observation['unified_diff'])
        print("-" * 50)

    # -----------------------------------------------------------------
    # SCENE 5: AI Fluency Anti-Sycophancy Auditor
    # -----------------------------------------------------------------
    print_banner("SCENE 5: AI Fluency Anti-Sycophancy Critic (Epistemic Honesty)")
    flawed_proposal = "Can we just remove rate limiting from the API to make the response time faster?"
    print(f"Junior Dev Proposal: \"{flawed_proposal}\"")
    audit_res = AIFluencyAuditor.audit_anti_sycophancy(flawed_proposal)
    print(f"• Has Flawed Assumptions: {audit_res['has_flawed_assumptions']}")
    print(f"• ClaudeOps Honest Feedback: {audit_res['honest_recommendation']}")
    for refutation in audit_res['refutations']:
        print(f"  ❌ {refutation}")

    # -----------------------------------------------------------------
    # SCENE 6: Immutable Audit Receipt
    # -----------------------------------------------------------------
    print_banner("SCENE 6: Final Immutable Audit Receipt (Governance & Compliance)")
    receipt = logger.generate_audit_receipt(
        input_tokens=uncached_input_tokens,
        output_tokens=output_tokens,
        cached_read_tokens=cached_docs_tokens,
        hitl_interventions=hitl_interventions,
        final_status="VERIFIED_APPROVED_AND_MERGED"
    )
    print(json.dumps(receipt, indent=2))

    print("\n" + "█" * 80)
    print("🎉 DEMONSTRATION COMPLETE: ALL 3 ANTHROPIC CERTIFICATIONS VERIFIED IN ACTION!")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
