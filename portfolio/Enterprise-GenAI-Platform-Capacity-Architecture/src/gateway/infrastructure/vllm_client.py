"""vLLM HTTP client — the only place in the codebase that talks to vLLM.

FastAPI routes NEVER import this directly. They go through InferenceService,
which owns lifecycle (slot acquisition, TTFT tracking, usage accounting).

Why isolate vLLM behind a client?
- Makes the mock backend swap-in trivial (same interface, different URL).
- Prevents tight coupling between request routing and inference protocol.
- Centralises timeout, retry, and error normalisation.

vLLM exposes an OpenAI-compatible API, so the client speaks that protocol.
In production, VLLM_BASE_URL points at the internal vLLM service (never public).
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger

logger = get_logger("vllm_client")


class VLLMError(Exception):
    """Raised when vLLM returns a non-2xx response or times out."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"vLLM error {status_code}: {detail}")


class VLLMClient:
    """Async HTTP client for the vLLM OpenAI-compatible inference API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._timeout = settings.inference_timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=float(self._timeout),
                write=10.0,
                pool=5.0,
            ),
        )
        logger.info("vllm_client_initialised", base_url=self._base_url)

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from vLLM /v1/models."""
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            return resp.json().get("data", [])
        except httpx.HTTPStatusError as e:
            raise VLLMError(e.response.status_code, e.response.text) from e
        except httpx.RequestError as e:
            raise VLLMError(503, f"vLLM unreachable: {e}") from e

    async def generate(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        """Non-streaming inference.

        Returns (response_json, ttft_ms).
        TTFT for non-streaming = time until the full response arrives.
        """
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            ttft_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            return resp.json(), ttft_ms
        except httpx.HTTPStatusError as e:
            raise VLLMError(e.response.status_code, e.response.text) from e
        except httpx.TimeoutException as e:
            raise VLLMError(504, "vLLM inference timeout") from e
        except httpx.RequestError as e:
            raise VLLMError(503, f"vLLM connection error: {e}") from e

    async def stream(
        self,
        payload: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Streaming inference — yields raw SSE lines from vLLM.

        The caller (InferenceService) is responsible for:
        - Measuring TTFT (time to first non-empty data line)
        - Handling client disconnects
        - Cleaning up concurrency slots in a finally block
        """
        stream_payload = {**payload, "stream": True}
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=stream_payload
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise VLLMError(resp.status_code, body.decode())

                async for line in resp.aiter_lines():
                    if line:
                        yield line
        except httpx.TimeoutException as e:
            raise VLLMError(504, "vLLM stream timeout") from e
        except httpx.RequestError as e:
            raise VLLMError(503, f"vLLM stream connection error: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()
        logger.info("vllm_client_closed")


# ── Singleton helper ───────────────────────────────────────────────────────────
# Shared client instance reuses the connection pool across all requests.

_client: VLLMClient | None = None


def get_vllm_client() -> VLLMClient:
    global _client
    if _client is None:
        _client = VLLMClient()
    return _client


async def close_vllm_client() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None
