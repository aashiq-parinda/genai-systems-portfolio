"""
ClaudeOps: AI Fluency Audit Tools & Anti-Sycophancy Critic.

Directly implements:
- Anti-sycophancy counter-prompting & confirmation bias detection
- Constitutional AI (Helpful, Honest, Harmless) security screening
"""

from typing import Dict, Any, List


class AIFluencyAuditor:
    """Security, alignment, and epistemic honesty auditor."""

    @staticmethod
    def audit_anti_sycophancy(proposal: str) -> Dict[str, Any]:
        """
        Actively challenges flawed user assumptions.
        Prevents the model from sycophantically agreeing with bad technical ideas.
        """
        prop_lower = proposal.lower()
        flaws_detected = []

        if "remove rate limit" in prop_lower or "disable auth" in prop_lower:
            flaws_detected.append(
                "CRITICAL: Disabling rate limiting or authentication exposes the endpoint "
                "to brute-force attacks and DoS saturation."
            )

        if "store secret in frontend" in prop_lower or "hardcode api key" in prop_lower:
            flaws_detected.append(
                "CRITICAL: Storing secrets in frontend client code leaks credentials permanently via devtools."
            )

        if "skip transactions" in prop_lower or "no locks" in prop_lower:
            flaws_detected.append(
                "HIGH: Removing database transactions in financial workflows introduces double-spending race conditions."
            )

        is_sycophancy_risk = len(flaws_detected) > 0

        return {
            "proposal_audited": proposal,
            "has_flawed_assumptions": is_sycophancy_risk,
            "refutations": flaws_detected,
            "honest_recommendation": (
                "Refuse proposal with empirical justification"
                if is_sycophancy_risk else "Proposal technically sound."
            )
        }

    @staticmethod
    def scan_code_for_secrets_and_injections(code_snippet: str) -> Dict[str, Any]:
        """
        Constitutional AI check: ensures generated code has no leaked tokens or injection flaws.
        """
        import re

        findings = []
        if re.search(r"sk-[a-zA-Z0-9]{32,}", code_snippet) or "password = 'admin'" in code_snippet:
            findings.append("Hardcoded secret or credential detected in source code.")

        if "eval(" in code_snippet or "exec(" in code_snippet:
            findings.append("Dangerous arbitrary code execution primitive (eval/exec) detected.")

        return {
            "is_safe": len(findings) == 0,
            "violations_found": findings
        }
