"""Inference orchestration service.

This is the core of the LLM API gateway. It owns:
  1. Concurrency slot acquisition (admission control)
  2. Prompt guardrail check (reuse EnterpriseGuardrails)
  3. TTFT measurement for streaming responses
  4. Client disconnect handling
  5. Timeout and error normalisation
  6. Concurrency slot release (always in finally)
  7. Usage event recording

Why this matters for LLM vs normal APIs:
  - Normal APIs: Request → DB query → Response (milliseconds, stateless)
  - LLM APIs:    Request → Acquire GPU slot → Send to inference → Stream tokens
                         → Release GPU slot → Record usage (seconds, stateful)

  The GPU slot lifecycle must be managed explicitly. A client disconnect
  mid-stream must still release the slot and record a partial usage event.
  This is fundamentally different from normal REST API design.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Optional

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from src.gateway.core.logging import get_logger
from src.gateway.infrastructure.vllm_client import VLLMClient, VLLMError
from src.gateway.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from src.gateway.services.model_router import GatewayModelRouter
from src.gateway.services.quota_service import QuotaService
from src.gateway.services.rate_limit_service import RateLimitService
from src.gateway.services.usage_service import UsageService

# Reuse the enterprise guardrails — no code duplication
from src.control_plane.guardrails import EnterpriseGuardrails

logger = get_logger("inference_service")

_guardrails = EnterpriseGuardrails()


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate for pre-check. ~1.3 tokens per word."""
    total_words = sum(len(m.content.split()) for m in messages)
    return max(1, int(total_words * 1.3))


class InferenceService:
    def __init__(
        self,
        vllm_client: VLLMClient,
        model_router: GatewayModelRouter,
        rate_limiter: RateLimitService,
        quota_service: QuotaService,
        usage_service: UsageService,
    ) -> None:
        self._client = vllm_client
        self._router = model_router
        self._rate_limiter = rate_limiter
        self._quota = quota_service
        self._usage = usage_service

    async def chat(
        self,
        req: ChatCompletionRequest,
        user_id: str,
        api_key_id: str,
        request_id: str,
    ):
        """Entry point for /v1/chat/completions. Handles both streaming and non-streaming."""
        # 1. Guardrail check — injection/PII scan using enterprise guardrails
        last_user_message = next(
            (m.content for m in reversed(req.messages) if m.role == "user"), ""
        )
        guard_result = _guardrails.inspect_prompt(last_user_message)
        if not guard_result.is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "safety_policy_violation",
                    "flags": guard_result.flags,
                },
            )

        # 2. Quota: load user's plan and check daily/monthly budgets
        from sqlalchemy.ext.asyncio import AsyncSession
        quota = await self._quota.get_quota_by_user_id(user_id)

        # 3. Rate limit: RPM check (already done in middleware; re-check for TPM)
        estimated_tokens = _estimate_tokens(req.messages)
        tpm_ok = await self._rate_limiter.check_tpm(user_id, estimated_tokens, quota)
        if not tpm_ok:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "tpm_limit_exceeded"},
                headers={"Retry-After": "60"},
            )

        # 4. Daily / monthly quota check
        await self._quota.check_daily_quota(user_id, quota)
        await self._quota.check_monthly_quota(user_id, quota)

        # 5. Concurrency slot — must be released in finally block
        slot_acquired = await self._rate_limiter.acquire_slot(user_id, quota)
        if not slot_acquired:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "concurrency_limit_exceeded",
                    "max_concurrent": quota.max_concurrent,
                },
                headers={"Retry-After": "10"},
            )

        # 6. Route to inference backend
        route = self._router.resolve(req.model, last_user_message)

        # 7. Build vLLM payload
        vllm_payload = {
            "model": route.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        if req.stop:
            vllm_payload["stop"] = req.stop

        if req.stream:
            return await self._handle_stream(
                vllm_payload, user_id, api_key_id, request_id, route
            )
        else:
            return await self._handle_non_stream(
                vllm_payload, user_id, api_key_id, request_id, route
            )

    async def _handle_non_stream(
        self, payload, user_id, api_key_id, request_id, route
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        try:
            response_json, ttft_ms = await self._client.generate(payload)
            latency_ms = (time.perf_counter() - t_start) * 1000

            usage = response_json.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            await self._quota.consume_tokens(user_id, input_tokens + output_tokens)
            await self._usage.record(
                request_id=request_id,
                user_id=user_id,
                api_key_id=api_key_id,
                model=route.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status_code=200,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                streaming=False,
            )
            return response_json

        except VLLMError as e:
            await self._usage.record(
                request_id=request_id,
                user_id=user_id,
                api_key_id=api_key_id,
                model=route.model_name,
                input_tokens=0,
                output_tokens=0,
                status_code=e.status_code,
                status_label="inference_error",
            )
            raise HTTPException(
                status_code=502,
                detail={"error": "inference_error", "detail": e.detail},
            )
        finally:
            await self._rate_limiter.release_slot(user_id)

    async def _handle_stream(
        self, payload, user_id, api_key_id, request_id, route
    ) -> StreamingResponse:
        """Return a StreamingResponse wrapping the async SSE generator."""

        async def sse_generator() -> AsyncGenerator[str, None]:
            t_start = time.perf_counter()
            ttft_ms: Optional[float] = None
            output_tokens = 0
            input_tokens = 0
            status_code = 200
            status_label = "success"

            try:
                async for raw_line in self._client.stream(payload):
                    if not raw_line.startswith("data:"):
                        continue

                    data_str = raw_line[5:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Measure TTFT — time from request start to first token byte
                    if ttft_ms is None:
                        delta_content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta_content:
                            ttft_ms = (time.perf_counter() - t_start) * 1000

                    # Count output tokens from usage field (present in final chunk)
                    if "usage" in chunk:
                        input_tokens = chunk["usage"].get("prompt_tokens", 0)
                        output_tokens = chunk["usage"].get("completion_tokens", 0)

                    # Count rough token estimate from delta content
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        output_tokens += max(1, len(content.split()))

                    yield f"data: {json.dumps(chunk)}\n\n"

            except VLLMError as e:
                status_code = e.status_code
                status_label = "inference_error"
                error_chunk = {"error": {"message": e.detail, "code": e.status_code}}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                status_code = 500
                status_label = "unexpected_error"
                logger.exception("stream_unexpected_error", request_id=request_id, error=str(e))
                yield "data: [DONE]\n\n"

            finally:
                # Slot release ALWAYS happens — even on client disconnect.
                # This is the most important lifecycle guarantee.
                await self._rate_limiter.release_slot(user_id)
                latency_ms = (time.perf_counter() - t_start) * 1000

                await self._quota.consume_tokens(user_id, input_tokens + output_tokens)
                await self._usage.record(
                    request_id=request_id,
                    user_id=user_id,
                    api_key_id=api_key_id,
                    model=route.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    status_code=status_code,
                    status_label=status_label,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    streaming=True,
                )

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
