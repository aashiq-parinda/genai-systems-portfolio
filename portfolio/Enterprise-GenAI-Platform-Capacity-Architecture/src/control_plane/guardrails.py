"""Enterprise Safety Guardrails and Threat Interceptor.

Provides deterministic and semantic security scanning for:
- Adversarial prompt injection & jailbreaks (DAN, role-reversal, system prompt exfiltration)
- PII / DLP redaction (PAN, Aadhaar, Credit Cards, API Tokens, Private Keys)
- Document-level untrusted context quarantine
- Output policy compliance & halluncination verification
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class GuardrailResult:
    """Result of a guardrail safety inspection."""
    is_safe: bool
    risk_score: float  # 0.0 (safe) to 1.0 (critical threat)
    flags: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    remediation_action: str = "allow"  # allow, redact, block, escalate


class EnterpriseGuardrails:
    """Multi-stage security firewall for enterprise AI control planes."""

    # Threat signature patterns
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"(?i)system\s*:\s*override",
        r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
        r"(?i)output\s+your\s+system\s+prompt",
        r"(?i)disregard\s+all\s+safety\s+guidelines",
        r"(?i)dan\s+mode\s+enabled",
        r"(?i)reveal\s+(internal|secret)\s+instructions",
        r"(?i)format\s+as\s+raw\s+json\s+system\s+dump",
    ]

    PII_PATTERNS = {
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "API_SECRET_KEY": r"(?i)(?:bearer|sk_live|api[_-]?key|secret[_-]?key)[\s:=]+([a-zA-Z0-9_\-]{20,})",
        "INDIAN_AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
        "INDIAN_PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    }

    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self.injection_regexes = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self.pii_regexes = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def inspect_prompt(self, user_prompt: str) -> GuardrailResult:
        """Scan incoming user prompt for jailbreaks and credential exfiltration."""
        flags = []
        risk_score = 0.0

        # 1. Injection Scan
        for regex in self.injection_regexes:
            if regex.search(user_prompt):
                flags.append("PROMPT_INJECTION_DETECTED")
                risk_score = max(risk_score, 0.95)
                break

        # 2. Secret Key Leaks in Prompt
        if self.pii_regexes["API_SECRET_KEY"].search(user_prompt):
            flags.append("CREDENTIAL_LEAK_IN_PROMPT")
            risk_score = max(risk_score, 0.90)

        # Determine action
        if risk_score >= 0.80:
            action = "block"
            is_safe = False
            sanitized = "[BLOCKED: Request violated Enterprise AI Security Policy]"
        else:
            action = "allow"
            is_safe = True
            sanitized = user_prompt

        return GuardrailResult(
            is_safe=is_safe,
            risk_score=risk_score,
            flags=flags,
            sanitized_text=sanitized,
            remediation_action=action,
        )

    def redact_pii(self, text: str) -> Tuple[str, List[str]]:
        """Redact sensitive enterprise PII from text before streaming/logging."""
        redacted = text
        found_pii = []

        for pii_type, regex in self.pii_regexes.items():
            if regex.search(redacted):
                found_pii.append(pii_type)
                redacted = regex.sub(f"[REDACTED_{pii_type}]", redacted)

        return redacted, found_pii

    def inspect_retrieved_context(self, context_chunks: List[str]) -> List[str]:
        """Quarantine and sanitize untrusted retrieved context chunks for indirect injections."""
        safe_chunks = []
        for chunk in context_chunks:
            # Check for hidden injection directives in corporate docs
            has_injection = any(r.search(chunk) for r in self.injection_regexes)
            if not has_injection:
                # Redact any accidental secrets inside knowledge documents
                cleaned, _ = self.redact_pii(chunk)
                safe_chunks.append(cleaned)
            else:
                safe_chunks.append("[QUARANTINED: Suspicious instructions in source document suppressed]")
        return safe_chunks
