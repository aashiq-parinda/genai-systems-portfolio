"""Mock vLLM inference server.

Simulates a vLLM OpenAI-compatible API endpoint for local development
and CI/CD. No GPU required.

What it simulates:
  - GET /v1/models — returns a list of fake models
  - POST /v1/chat/completions — returns a streamed or non-streamed response
    with realistic token-by-token timing
  - Approximate token counting to make usage accounting realistic

Educational value:
  - Lets developers experience the full streaming protocol without a GPU.
  - TTFT, inter-token latency, and [DONE] frame are all simulated.
  - The gateway's inference lifecycle code runs identically against this
    server as it would against a real vLLM node.

IMPORTANT: This is NOT production inference. It returns pre-scripted tokens.
"""

import asyncio
import json
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Mock vLLM Inference Server", version="1.0.0")

# ── Schemas ────────────────────────────────────────────────────────────────────

class MockMessage(BaseModel):
    role: str
    content: str


class MockChatRequest(BaseModel):
    model: str = "local-model"
    messages: list[MockMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: bool = False


# ── Simulated response content ─────────────────────────────────────────────────

_RESPONSES = {
    "kv cache": (
        "KV cache stores the key-value tensors from the attention mechanism "
        "for each token that has already been processed. During autoregressive "
        "decoding, instead of recomputing attention over the entire context on "
        "every step, the model reads the cached KV tensors. This reduces "
        "per-step compute from O(n²) to O(n), making streaming generation "
        "practical at long context lengths. The trade-off is VRAM: each "
        "active stream holds KV state proportional to its context length."
    ),
    "continuous batching": (
        "Continuous batching (also called iteration-level scheduling) allows "
        "vLLM to insert new requests into a running batch mid-generation. "
        "Traditional static batching waits for all sequences to finish before "
        "accepting new work, wasting GPU time when some sequences finish early. "
        "Continuous batching keeps the GPU saturated by immediately filling "
        "freed slots with new requests, dramatically improving throughput."
    ),
    "default": (
        "This is a simulated response from the mock vLLM inference backend. "
        "In production, replace INFERENCE_BACKEND=mock with INFERENCE_BACKEND=vllm "
        "and set VLLM_BASE_URL to your vLLM node. The gateway code is identical "
        "in both cases — only the backend URL changes."
    ),
}


def _select_response(messages: list[MockMessage]) -> str:
    last = next((m.content.lower() for m in reversed(messages) if m.role == "user"), "")
    for keyword, response in _RESPONSES.items():
        if keyword in last:
            return response
    return _RESPONSES["default"]


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _count_prompt_tokens(messages: list[MockMessage]) -> int:
    return sum(_count_tokens(m.content) for m in messages)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "local-model", "object": "model", "owned_by": "mock"},
            {"id": "fast-model", "object": "model", "owned_by": "mock"},
            {"id": "fallback-model", "object": "model", "owned_by": "mock"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: MockChatRequest):
    content = _select_response(req.messages)
    words = content.split()
    prompt_tokens = _count_prompt_tokens(req.messages)
    completion_tokens = len(words)
    chat_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_response(chat_id, created, req.model, words, prompt_tokens),
            media_type="text/event-stream",
        )

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_response(chat_id, created, model, words, prompt_tokens):
    """Simulate token-by-token streaming with realistic inter-token delay.

    TTFT ≈ 150ms (simulated GPU scheduling overhead)
    Inter-token latency ≈ 30ms (simulated at ~33 tokens/sec for an 8B model)
    """
    # First chunk: role delta (marks start of stream — TTFT measured here)
    await asyncio.sleep(0.15)  # simulated TTFT
    first = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    # Token stream
    for word in words:
        await asyncio.sleep(0.03)  # ~33 tokens/sec inter-token latency
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with usage
    final = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(words),
            "total_tokens": prompt_tokens + len(words),
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mock-inference"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
