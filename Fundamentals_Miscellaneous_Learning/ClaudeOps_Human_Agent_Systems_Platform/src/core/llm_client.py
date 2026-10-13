"""
ClaudeOps: Dual-Mode LLM Client (Seamless Mock vs Real Anthropic API).

Auto-detects ANTHROPIC_API_KEY:
- If key present: Calls official Anthropic Messages API with Prompt Caching.
- If key absent: Uses high-fidelity, deterministic local simulator (zero cost, zero latency).
"""

import os
import json
from typing import Dict, Any, List, Optional
from config import ClaudeModelTier


class DualModeClaudeClient:
    """Unified client supporting both zero-cost mock execution and real Anthropic API calls."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.is_live = bool(self.api_key and self.api_key.startswith("sk-ant-"))
        self._sdk_client = None

        if self.is_live:
            try:
                import anthropic
                self._sdk_client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  'anthropic' library not installed. Falling back to Mock Mode.")
                self.is_live = False

    def is_mock_mode(self) -> bool:
        """Returns True if running in zero-cost mock mode."""
        return not self.is_live

    def create_message(
        self,
        model: ClaudeModelTier,
        system: Any,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes a completion turn either against the live API or mock engine.
        """
        if self.is_live and self._sdk_client:
            response = self._sdk_client.messages.create(
                model=model.value,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages
            )
            # Format live response
            content_text = "".join(b.text for b in response.content if hasattr(b, "text"))
            return {
                "mode": "LIVE_API",
                "model": model.value,
                "content": content_text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }
            }

        # -------------------------------------------------------------
        # Deterministic Mock Engine (Course 1, 2, 3 Simulation)
        # -------------------------------------------------------------
        user_text = messages[-1].get("content", "")
        if isinstance(user_text, list):
            user_text = " ".join(item.get("text", "") for item in user_text if isinstance(item, dict))

        # Check for prefill turn
        has_prefill = len(messages) > 1 and messages[-1].get("role") == "assistant"
        prefill_text = messages[-1].get("content", "") if has_prefill else ""

        if "timing attack" in user_text.lower() or "auth" in user_text.lower():
            mock_content = (
                "<thinking>\n"
                "1. Vulnerability Analysis: String equality (==) short-circuits on byte mismatch, leaking timing information.\n"
                "2. Remediation Strategy: Use hmac.compare_digest() for constant-time comparison.\n"
                "3. In addition, inject token bucket check_rate_limit(username) to prevent brute-force attacks.\n"
                "4. Constraint Check: All imported modules (hmac) are Python standard library.\n"
                "</thinking>\n\n"
                "```python\n"
                "import hmac\n"
                "from src.core.rate_limiter import check_rate_limit\n\n"
                "def authenticate(username, password):\n"
                "    check_rate_limit(username)\n"
                "    if hmac.compare_digest(password, 'admin_secret'):\n"
                "        return {'token': 'jwt_mock_token_123'}\n"
                "    return None\n"
                "```"
            )
        else:
            mock_content = (
                "<thinking>\nAnalyzing request constraints and constructing verified patch.\n</thinking>\n"
                "Patch generated successfully conforming to all negative constraints."
            )

        if has_prefill and prefill_text == "{":
            mock_content = ' "status": "VERIFIED_SECURE", "patch_applied": true, "timing_attack_remediated": true }'

        return {
            "mode": "MOCK_SIMULATOR",
            "model": model.value,
            "content": mock_content,
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 850,
                "cache_read_input_tokens": 45000  # Simulating 90% prompt caching
            }
        }
