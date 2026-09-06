"""Model router — gateway adapter over the enterprise DynamicModelRouter.

This is the integration point between the public API gateway and the
existing enterprise control plane router. Both systems share the same
complexity-scoring logic, demonstrating architectural reuse.

Enterprise control plane: routes by bot policy (force_frontier, dynamic_routed)
Public API gateway: routes by model name string from the client request

Future routing dimensions this interface can accommodate:
  - GPU availability / queue depth (from vLLM /metrics)
  - Per-model latency SLOs
  - Cost-aware load balancing
  - Capacity-based admission (reject when all GPUs are saturated)
"""

from dataclasses import dataclass
from typing import Optional

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger

# Reuse the enterprise router — no code duplication
from src.control_plane.router import DynamicModelRouter, RouteDecision

logger = get_logger("model_router")


@dataclass
class GatewayRouteDecision:
    model_name: str          # resolved model identifier
    backend_url: str         # internal inference backend URL
    tier: str                # "SLM" | "FRONTIER" | "MOCK"
    complexity_score: float
    estimated_cost_usd: float
    routing_reason: str


# Model name → backend URL mapping.
# In production this would be loaded from a service registry or config file.
_DEFAULT_MODEL_REGISTRY: dict[str, str] = {
    "local-model": "{vllm_url}",
    "fast-model": "{vllm_url}",
    "fallback-model": "{vllm_url}",
}


class GatewayModelRouter:
    """Routes API gateway requests to the appropriate inference backend.

    Wraps DynamicModelRouter from the enterprise control plane for complexity
    scoring. The gateway adds model-name resolution and backend URL mapping.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._enterprise_router = DynamicModelRouter()
        self._registry = {
            name: url.format(vllm_url=self._settings.vllm_base_url)
            for name, url in _DEFAULT_MODEL_REGISTRY.items()
        }

    def list_models(self) -> list[dict]:
        """Return models available in this gateway."""
        return [
            {"id": name, "object": "model", "owned_by": "local"}
            for name in self._registry
        ]

    def resolve(
        self,
        model_name: str,
        prompt: str = "",
        force_frontier: bool = False,
    ) -> GatewayRouteDecision:
        """Resolve a client model name to an inference backend URL.

        Falls back to "local-model" if the requested model isn't registered.
        """
        if model_name not in self._registry:
            logger.warning(
                "unknown_model_requested",
                model=model_name,
                fallback="local-model",
            )
            model_name = "local-model"

        # Use the enterprise router's complexity scoring for routing metadata.
        # In this educational implementation, all models map to the same vLLM URL.
        # A production system would use this score to route between different GPU pools.
        enterprise_decision: RouteDecision = self._enterprise_router.route_request(
            prompt=prompt,
            force_frontier=force_frontier,
        )

        backend_url = self._registry[model_name]
        tier = "MOCK" if self._settings.inference_backend == "mock" else enterprise_decision.tier

        return GatewayRouteDecision(
            model_name=model_name,
            backend_url=backend_url,
            tier=tier,
            complexity_score=enterprise_decision.complexity_score,
            estimated_cost_usd=enterprise_decision.estimated_cost_usd,
            routing_reason=enterprise_decision.routing_reason,
        )
