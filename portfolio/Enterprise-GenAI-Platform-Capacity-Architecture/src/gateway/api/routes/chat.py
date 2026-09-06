"""Chat completion route — the primary LLM inference endpoint."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.infrastructure.database import get_db
from src.gateway.infrastructure.vllm_client import get_vllm_client
from src.gateway.schemas.chat import ChatCompletionRequest
from src.gateway.services.inference_service import InferenceService
from src.gateway.services.model_router import GatewayModelRouter
from src.gateway.services.quota_service import QuotaService
from src.gateway.services.rate_limit_service import RateLimitService
from src.gateway.services.usage_service import UsageService

router = APIRouter(prefix="/v1", tags=["Chat"])

# Singleton router — shared across all requests in this process
_model_router = GatewayModelRouter()


async def _get_inference_service(db: AsyncSession = Depends(get_db)) -> InferenceService:
    """Build the inference service with all dependencies injected."""
    return InferenceService(
        vllm_client=get_vllm_client(),
        model_router=_model_router,
        rate_limiter=RateLimitService(),
        quota_service=QuotaService(db),
        usage_service=UsageService(db),
    )


@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    service: InferenceService = Depends(_get_inference_service),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (`stream=true`, Server-Sent Events) and
    non-streaming responses. Rate limiting, token quotas, concurrency
    management, guardrails, and usage accounting are all applied automatically.

    ```bash
    curl http://localhost:8001/v1/chat/completions \\
      -H "Authorization: Bearer llm_xxx" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "local-model",
        "messages": [{"role": "user", "content": "Explain KV cache."}],
        "max_tokens": 300,
        "stream": false
      }'
    ```
    """
    user = request.state.user
    api_key = request.state.api_key

    return await service.chat(
        req=req,
        user_id=user.id,
        api_key_id=api_key.id,
        request_id=request.state.request_id,
    )


@router.post("/embeddings")
async def embeddings():
    """
    Embeddings endpoint (stub).

    > [Educational Note] In production, this would forward to a dedicated
    embedding model (e.g. BGE, E5) running on a separate inference node.
    Mixing embedding and generative workloads on the same GPU wastes
    expensive batch capacity. See docs/07_API_GATEWAY_LAYER.md for details.
    """
    return {
        "error": "not_implemented",
        "detail": (
            "Embeddings require a dedicated embedding model endpoint. "
            "See INFERENCE_BACKEND docs to connect a real embedding service."
        ),
    }
