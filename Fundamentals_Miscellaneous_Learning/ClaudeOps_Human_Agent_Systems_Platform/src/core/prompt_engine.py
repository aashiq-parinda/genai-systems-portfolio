"""
ClaudeOps: Prompt Engine & Anthropic Formatting Architecture.

Implements core practices from 'Claude Platform 101':
- Canonical XML tag formatting (<system>, <context>, <instructions>, <thinking>, <output>)
- Scratchpad / Chain-of-Thought extraction (<thinking>)
- Assistant Response Prefilling (for deterministic JSON / structured tags)
- Prompt Caching payload construction with 'cache_control': {'type': 'ephemeral'}
- Anti-Sycophancy prompt injection for epistemic honesty
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from config import ClaudeModelTier


class ClaudePromptEngine:
    """Enterprise prompt construction and response parsing engine tailored for Claude."""

    @staticmethod
    def format_xml_prompt(
        system_role: str,
        instructions: List[str],
        user_input: str,
        context: Optional[str] = None,
        negative_constraints: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        enable_thinking_scratchpad: bool = True
    ) -> str:
        """
        Builds a canonical XML prompt following Anthropic's pre-training structure.
        """
        parts = []

        # 1. System Role & Persona
        parts.append(f"<system>\n{system_role.strip()}\n</system>")

        # 2. Domain Context & Documentation
        if context:
            parts.append(f"<context>\n{context.strip()}\n</context>")

        # 3. Operational Instructions
        inst_list = list(instructions)
        if enable_thinking_scratchpad:
            inst_list.append(
                "Wrap your detailed intermediate step-by-step reasoning inside <thinking> tags "
                "before producing the final deliverable."
            )
        formatted_instructions = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(inst_list))
        parts.append(f"<instructions>\n{formatted_instructions}\n</instructions>")

        # 4. Explicit Negative Constraints (Anthropic Best Practice: What NOT to do)
        if negative_constraints:
            neg_str = "\n".join(f"- {c}" for c in negative_constraints)
            parts.append(f"<negative_constraints>\n{neg_str}\n</negative_constraints>")

        # 5. Few-Shot Canonical Examples
        if examples:
            ex_blocks = ["<examples>"]
            for ex in examples:
                ex_blocks.append("  <example>")
                ex_blocks.append(f"    <input>{ex.get('input', '').strip()}</input>")
                ex_blocks.append(f"    <ideal_output>{ex.get('output', '').strip()}</ideal_output>")
                ex_blocks.append("  </example>")
            ex_blocks.append("</examples>")
            parts.append("\n".join(ex_blocks))

        # 6. Dynamic User Input
        parts.append(f"<user_input>\n{user_input.strip()}\n</user_input>")

        return "\n\n".join(parts)

    @staticmethod
    def extract_tag(content: str, tag_name: str) -> Optional[str]:
        """
        Safely extracts inner text from an XML tag (<tag>...</tag>).
        Uses non-greedy matching across newlines.
        """
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_scratchpad_and_answer(response_text: str) -> Tuple[Optional[str], str]:
        """
        Separates the internal <thinking> scratchpad from the user-facing answer.
        Protects the user from cognitive overload while preserving the reasoning trail.
        """
        thinking = ClaudePromptEngine.extract_tag(response_text, "thinking")
        # Remove the <thinking> block to leave the clean answer
        cleaned_answer = re.sub(r"<thinking>.*?</thinking>", "", response_text, flags=re.DOTALL).strip()
        return thinking, cleaned_answer

    @staticmethod
    def build_cached_payload(
        model: ClaudeModelTier,
        system_prompt: str,
        static_context: str,
        dynamic_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Builds an Anthropic Messages API payload with Prompt Caching.
        Applies 'cache_control': {'type': 'ephemeral'} to the static system & context blocks
        to unlock the 90% read cost discount on subsequent calls within the 5-minute TTL.
        """
        system_role_block = system_prompt.strip()
        return {
            "model": model.value,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [
                {
                    "type": "text",
                    "text": system_role_block
                },
                {
                    "type": "text",
                    "text": static_context.strip(),
                    "cache_control": {"type": "ephemeral"}  # Anthropic Prompt Caching anchor
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": dynamic_prompt.strip()
                }
            ]
        }

    @staticmethod
    def prepare_prefill_turn(
        messages: List[Dict[str, str]],
        prefill_text: str = "{"
    ) -> List[Dict[str, str]]:
        """
        Adds an assistant prefill turn to enforce immediate JSON or XML formatting.
        Eliminates 'Sure! Here is the JSON:' conversational filler.
        """
        prepared = list(messages)
        prepared.append({
            "role": "assistant",
            "content": prefill_text
        })
        return prepared

    @staticmethod
    def parse_prefilled_json(raw_completion: str, prefill_prefix: str = "{") -> Dict[str, Any]:
        """
        Parses a completion that continued from a prefilled opening brace.
        """
        text = raw_completion.strip()
        if not text.startswith("{") and prefill_prefix == "{":
            text = "{" + text

        # Clean trailing backticks if any
        if text.endswith("```"):
            text = text[:-3].strip()

        return json.loads(text)
