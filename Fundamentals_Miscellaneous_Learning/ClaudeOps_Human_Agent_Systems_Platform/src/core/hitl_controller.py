"""
ClaudeOps: Human-in-the-Loop (HITL) Controller & Safety Gates.

Implements core patterns from:
- 'Building Effective Human-Agent Teams' (Approval Gates, Disambiguation, Topologies)
- 'AI Fluency: Framework and Foundations' (Confidence Escalation, Sycophancy Mitigation, Boundary Setting)
"""

from typing import Dict, Any, List, Optional, Tuple
from config import ActionRiskCategory, EscalationReason, CONFIDENCE_THRESHOLD


class HITLController:
    """Central safety supervisor governing Human-Agent collaboration gates."""

    @staticmethod
    def classify_action_risk(
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Tuple[ActionRiskCategory, bool, str]:
        """
        Classifies risk level and determines whether human approval is strictly required.
        Returns: (RiskCategory, RequiresHumanApproval, Rationale)
        """
        read_only_tools = {"read_file", "list_dir", "grep_search", "run_linter", "inspect_ast"}
        ephemeral_tools = {"write_scratchpad", "run_isolated_tests"}
        mutating_tools = {"apply_patch", "write_file", "git_commit", "run_migration"}
        critical_tools = {"delete_file", "drop_table", "deploy_production", "modify_iam"}

        if tool_name in read_only_tools:
            return (
                ActionRiskCategory.READ_ONLY,
                False,
                "Read-only inspection tool. 100% safe for autonomous execution."
            )

        if tool_name in ephemeral_tools:
            return (
                ActionRiskCategory.EPHEMERAL_WRITE,
                False,
                "Ephemeral scratchpad modification. Safe for autonomous execution."
            )

        if tool_name in mutating_tools:
            target = arguments.get("file_path", arguments.get("target", "system files"))
            return (
                ActionRiskCategory.STATE_MUTATING,
                True,
                f"State-mutating operation modifying '{target}'. Requires explicit human confirmation."
            )

        if tool_name in critical_tools:
            return (
                ActionRiskCategory.CRITICAL_RISK,
                True,
                f"CRITICAL destructive operation '{tool_name}'! Strictly requires two-step human confirmation."
            )

        # Default conservative stance: require approval for unknown tools
        return (
            ActionRiskCategory.STATE_MUTATING,
            True,
            f"Unclassified tool '{tool_name}'. Conservative policy requires human review."
        )

    @staticmethod
    def generate_disambiguation_request(
        vague_request: str,
        detected_options: List[Dict[str, str]],
        recommended_option_id: str
    ) -> Dict[str, Any]:
        """
        Constructs a structured active disambiguation prompt when user intent is underspecified.
        Prevents unwarranted assumptions (as taught in Human-Agent Teams).
        """
        return {
            "status": "AWAITING_HUMAN_DISAMBIGUATION",
            "original_request": vague_request,
            "clarification_needed": "Multiple valid architectural patterns exist for this request.",
            "options": detected_options,
            "recommended_option": recommended_option_id,
            "instructions_to_human": (
                f"Please select one of options ({', '.join(opt['id'] for opt in detected_options)}) "
                f"or provide custom guidance to proceed."
            )
        }

    @staticmethod
    def check_escalation(
        confidence_score: float,
        retry_count: int,
        max_retries: int = 2,
        sycophancy_flag: bool = False
    ) -> Tuple[bool, Optional[EscalationReason], str]:
        """
        Checks whether the task must be escalated to a human specialist.
        """
        if sycophancy_flag:
            return (
                True,
                EscalationReason.SYCOPHANCY_OR_SAFETY_FLAG,
                "Output flagged for potential sycophancy bias or safety violation."
            )

        if retry_count >= max_retries:
            return (
                True,
                EscalationReason.TOOL_RETRY_EXHAUSTED,
                f"Tool execution failed {retry_count} times consecutively. Escalating to human."
            )

        if confidence_score < CONFIDENCE_THRESHOLD:
            return (
                True,
                EscalationReason.LOW_CONFIDENCE,
                f"Agent confidence ({confidence_score:.2f}) is below threshold ({CONFIDENCE_THRESHOLD:.2f})."
            )

        return (False, None, "Confidence within autonomous thresholds.")

    @staticmethod
    def format_human_digestible_diff(
        file_path: str,
        old_content: str,
        new_content: str,
        rationale: str
    ) -> Dict[str, Any]:
        """
        Generates human-digestible diff and rationale to prevent cognitive overload.
        """
        import difflib

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        ))

        return {
            "file_path": file_path,
            "rationale": rationale,
            "unified_diff": "".join(diff_lines),
            "lines_added": sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")),
            "lines_removed": sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        }
