"""
LangChain Core: Memory, State History & Message Trimming Patterns.
Demonstrates:
1. Message Roles (System, Human, AI, Tool)
2. Sliding Window Buffer (Token / Message Trim)
3. Rolling Summarization Memory
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "human", "ai", "tool"]
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationBuffer:
    """Manages raw in-memory chat history for LangChain chains."""
    def __init__(self):
        self.messages: List[ChatMessage] = []

    def add_user_message(self, text: str):
        self.messages.append(ChatMessage(role="human", content=text))

    def add_ai_message(self, text: str):
        self.messages.append(ChatMessage(role="ai", content=text))

    def add_system_message(self, text: str):
        self.messages.insert(0, ChatMessage(role="system", content=text))

    def get_messages(self) -> List[ChatMessage]:
        return list(self.messages)

    def clear(self):
        self.messages.clear()

def trim_messages(messages: List[ChatMessage], max_messages: int = 4, keep_system: bool = True) -> List[ChatMessage]:
    """
    Sliding window message trimmer.
    Preserves the system message while keeping only the most recent N turns to prevent context window blowup.
    """
    if not messages:
        return []

    system_msg = None
    dialogue = []
    for msg in messages:
        if msg.role == "system" and system_msg is None and keep_system:
            system_msg = msg
        else:
            dialogue.append(msg)

    trimmed_dialogue = dialogue[-max_messages:]
    if system_msg:
        return [system_msg] + trimmed_dialogue
    return trimmed_dialogue

class SummaryMemoryManager:
    """
    Progressively compiles older chat turns into a running summary to minimize token usage.
    """
    def __init__(self):
        self.running_summary: str = "No prior discussion."
        self.active_turns: List[ChatMessage] = []

    def add_interaction(self, user_text: str, ai_text: str):
        self.active_turns.append(ChatMessage(role="human", content=user_text))
        self.active_turns.append(ChatMessage(role="ai", content=ai_text))

        # When active turns exceed 4, condense older turns into the summary
        if len(self.active_turns) > 4:
            condensed_turn = self.active_turns.pop(0)
            condensed_response = self.active_turns.pop(0)
            self.running_summary += f" [User asked '{condensed_turn.content[:25]}...', AI provided solution]."

    def get_prompt_context(self) -> str:
        active_repr = "\n".join([f"{m.role.upper()}: {m.content}" for m in self.active_turns])
        return (
            f"=== Running Conversation Summary ===\n{self.running_summary}\n\n"
            f"=== Recent Turns ===\n{active_repr}"
        )
