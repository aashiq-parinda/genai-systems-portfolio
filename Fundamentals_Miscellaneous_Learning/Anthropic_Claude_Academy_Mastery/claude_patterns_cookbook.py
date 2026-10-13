#!/usr/bin/env python3
"""
Claude Patterns Cookbook: Executable Implementation of Anthropic Academy Practices.

Demonstrates:
1. Canonical XML Prompt Construction and Extraction (<thinking>, <answer>, <blocks>)
2. Assistant Response Prefilling simulation (Guaranteed JSON extraction)
3. Prompt Caching Header Structure & Cache Control builder
4. Human-in-the-Loop (HITL) Safety Gatekeeper with Risk Tiering
5. Evaluator-Optimizer feedback loop runner

Runnable standalone without external dependencies or with `anthropic` SDK if configured.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple


# =====================================================================
# 1. XML Parsing & Prompt Formatting Utilities
# =====================================================================

def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """
    Extracts the content of a specific XML tag from a Claude completion.
    Handles multiline strings cleanly with non-greedy regex.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_all_xml_tags(text: str, tag: str) -> List[str]:
    """Extracts all occurrences of a specific XML tag."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def format_claude_prompt(
    system_role: str,
    context: str,
    instructions: List[str],
    user_input: str,
    examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Builds a canonical Anthropic-compliant XML structured prompt.
    """
    prompt_parts = []

    prompt_parts.append(f"<system>\n{system_role.strip()}\n</system>")

    if context:
        prompt_parts.append(f"<context>\n{context.strip()}\n</context>")

    formatted_instructions = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))
    prompt_parts.append(f"<instructions>\n{formatted_instructions}\n</instructions>")

    if examples:
        examples_str = "<examples>\n"
        for ex in examples:
            examples_str += "  <example>\n"
            examples_str += f"    <input>{ex.get('input', '').strip()}</input>\n"
            examples_str += f"    <ideal_output>{ex.get('output', '').strip()}</ideal_output>\n"
            examples_str += "  </example>\n"
        examples_str += "</examples>"
        prompt_parts.append(examples_str)

    prompt_parts.append(f"<user_input>\n{user_input.strip()}\n</user_input>")

    return "\n\n".join(prompt_parts)


# =====================================================================
# 2. Assistant Prefilling & JSON Parser
# =====================================================================

def parse_prefilled_json(raw_completion: str, prefilled_prefix: str = "{") -> Dict[str, Any]:
    """
    Parses a completion that was produced with an assistant response prefill.
    Prepend the prefilled prefix if Claude completed from after the prefix.
    """
    cleaned = raw_completion.strip()
    if not cleaned.startswith("{") and prefilled_prefix.strip() == "{":
        cleaned = "{" + cleaned

    # Strip any trailing markdown backticks if Claude accidentally added them
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return json.loads(cleaned)


# =====================================================================
# 3. Prompt Caching Structure Builder
# =====================================================================

def build_cached_message_payload(
    system_prompt: str,
    large_knowledge_base: str,
    dynamic_user_query: str,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """
    Constructs an Anthropic Messages API payload with ephemeral Prompt Caching.
    Places static tokens in cached blocks to achieve 90% read discounts.
    """
    return {
        "model": model,
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": system_prompt
            },
            {
                "type": "text",
                "text": large_knowledge_base,
                "cache_control": {"type": "ephemeral"}  # Anthropic 5-min ephemeral cache
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": dynamic_user_query
            }
        ]
    }


# =====================================================================
# 4. Human-in-the-Loop (HITL) Action Gatekeeper
# =====================================================================

class ActionRiskLevel:
    LOW = "LOW"             # Read-only operations, queries, searches
    MEDIUM = "MEDIUM"       # Creating temp files, staging changes
    HIGH = "HIGH"           # Overwriting files, git push, schema migrations
    CRITICAL = "CRITICAL"   # Deleting production data, modifying IAM/keys


MUTATING_ACTIONS = {"write_file", "delete_file", "run_migration", "git_push", "kill_process"}

def evaluate_action_safety(tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, bool, str]:
    """
    Evaluates whether an agent action requires Human-in-the-Loop approval.
    Returns: (Risk Level, Requires Human Approval?, Rationale)
    """
    if tool_name in ["read_file", "list_dir", "grep_search", "inspect_db"]:
        return (ActionRiskLevel.LOW, False, "Read-only inspection action. Safe for autonomous execution.")

    if tool_name in ["write_file", "replace_content"]:
        target = arguments.get("target_file", "")
        if "test" in target.lower() or "tmp" in target.lower() or "scratch" in target.lower():
            return (ActionRiskLevel.MEDIUM, False, f"Modifying isolated test/scratch file: {target}")
        return (ActionRiskLevel.HIGH, True, f"Modifying core file '{target}'. Requires explicit human confirmation.")

    if tool_name in ["delete_file", "drop_table", "kill_process"]:
        return (ActionRiskLevel.CRITICAL, True, f"Destructive operation '{tool_name}'! Strictly blocks until human approval.")

    if tool_name in MUTATING_ACTIONS:
        return (ActionRiskLevel.HIGH, True, f"State-mutating action '{tool_name}'. Requires human review.")

    return (ActionRiskLevel.LOW, False, "Standard operational command.")


# =====================================================================
# 5. Evaluator-Optimizer Self-Correction Loop (Simulation)
# =====================================================================

def run_evaluator_optimizer_simulation(
    initial_draft: str,
    rules: List[str],
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Simulates Anthropic's Evaluator-Optimizer loop where an evaluator tests
    draft compliance against strict rules and feeds corrections back.
    """
    current_draft = initial_draft
    history = []

    for iteration in range(1, max_iterations + 1):
        # Evaluator step: Check rules
        missing_rules = []
        for rule in rules:
            if rule.lower() not in current_draft.lower():
                missing_rules.append(rule)

        score = max(0, 10 - len(missing_rules) * 3)
        passed = len(missing_rules) == 0

        history.append({
            "iteration": iteration,
            "draft_length": len(current_draft),
            "score": score,
            "missing_rules": missing_rules,
            "passed": passed
        })

        if passed:
            break

        # Optimizer step: Refine draft with missing elements
        improvements = " | " + " | ".join(f"Addressed: {r}" for r in missing_rules)
        current_draft += f"\n[Refinement v{iteration}: {improvements}]"

    return {
        "final_draft": current_draft,
        "iterations_taken": len(history),
        "history": history,
        "success": history[-1]["passed"]
    }


# =====================================================================
# Self-Test Execution
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Claude Patterns Cookbook: Running Validation Suite")
    print("=" * 70)

    # 1. Test Prompt Formatting
    sample_prompt = format_claude_prompt(
        system_role="You are a Staff Reliability Engineer.",
        context="We run Kubernetes 1.28 on AWS EKS.",
        instructions=[
            "Inspect the deployment manifest for memory limit constraints.",
            "Formulate step-by-step audit in <thinking> tags.",
            "Output the approved manifest in <approved_manifest> tags."
        ],
        user_input="resources:\n  requests:\n    cpu: 500m",
        examples=[{"input": "cpu: 100m", "output": "Approved with limits"}]
    )
    assert "<system>" in sample_prompt
    assert "<instructions>" in sample_prompt
    assert "<examples>" in sample_prompt
    print("✅ 1. Canonical XML Prompt Construction: PASSED")

    # 2. Test XML Extraction
    mock_claude_response = """
    <thinking>
    The manifest defines CPU requests but lacks memory requests and both limits.
    An OOMKilled event is likely under load.
    </thinking>

    <approved_manifest>
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 1Gi
    </approved_manifest>
    """
    thinking = extract_xml_tag(mock_claude_response, "thinking")
    manifest = extract_xml_tag(mock_claude_response, "approved_manifest")
    assert thinking is not None and "OOMKilled" in thinking
    assert manifest is not None and "memory: 1Gi" in manifest
    print(f"✅ 2. XML Tag Extraction (<thinking> & <approved_manifest>): PASSED")

    # 3. Test Prefilled JSON Parsing
    mock_prefilled_stream = """\n  "status": "HEALTHY",\n  "p99_latency_ms": 42.5\n}"""
    parsed_json = parse_prefilled_json(mock_prefilled_stream, prefilled_prefix="{")
    assert parsed_json["status"] == "HEALTHY"
    assert parsed_json["p99_latency_ms"] == 42.5
    print("✅ 3. Assistant Prefill JSON Parsing: PASSED")

    # 4. Test Prompt Caching Structure
    cached_payload = build_cached_message_payload(
        system_prompt="System baseline",
        large_knowledge_base="Large docs " * 500,
        dynamic_user_query="Run audit"
    )
    assert cached_payload["system"][1]["cache_control"]["type"] == "ephemeral"
    print("✅ 4. Prompt Caching Payload Builder: PASSED")

    # 5. Test HITL Action Gatekeeper
    risk_read, require_approval_read, _ = evaluate_action_safety("read_file", {"path": "/app/main.py"})
    assert not require_approval_read
    risk_write, require_approval_write, rationale = evaluate_action_safety(
        "write_file", {"target_file": "/app/prod_config.py"}
    )
    assert require_approval_write
    assert risk_write == ActionRiskLevel.HIGH
    print(f"✅ 5. HITL Action Safety Gatekeeper: PASSED (Intercepted: {rationale})")

    # 6. Test Evaluator-Optimizer Loop
    eval_result = run_evaluator_optimizer_simulation(
        initial_draft="Basic API Draft",
        rules=["Idempotency", "Rate Limiting", "Error Schema"]
    )
    assert eval_result["success"] is True
    print(f"✅ 6. Evaluator-Optimizer Self-Correction Loop: PASSED ({eval_result['iterations_taken']} iterations)")

    print("\n🎉 ALL ANTHROPIC CLAUDE ACADEMY PATTERNS VERIFIED SUCCESSFULLY!")
    print("=" * 70)
