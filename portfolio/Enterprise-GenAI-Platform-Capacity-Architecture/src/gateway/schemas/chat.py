"""Chat completion schemas — OpenAI-compatible.

We implement the same wire format as OpenAI so existing clients (LangChain,
LlamaIndex, litellm) can point at this gateway with a simple base_url change.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="local-model", description="Model identifier")
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None

    model_config = {"extra": "ignore"}  # forward-compatible with newer OpenAI params


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    """Non-streaming response. OpenAI-compatible."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


# ── Streaming ──────────────────────────────────────────────────────────────────

class ChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatStreamChoice(BaseModel):
    index: int
    delta: ChatDelta
    finish_reason: Optional[str] = None


class ChatStreamChunk(BaseModel):
    """One SSE data frame. OpenAI-compatible."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamChoice]
