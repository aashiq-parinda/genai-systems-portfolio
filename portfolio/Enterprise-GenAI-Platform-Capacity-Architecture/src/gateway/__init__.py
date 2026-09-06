"""
Gateway sub-package for the public API key-based LLM API gateway.

This package sits alongside src/control_plane/ (the enterprise multi-tenant
control plane) and shares its DynamicModelRouter and EnterpriseGuardrails,
but exposes a different auth model (API keys vs JWT) and adds:
  - Redis-backed distributed rate limiting
  - Token-based quota enforcement
  - Durable PostgreSQL usage accounting
  - Risk/identity scoring
  - Prometheus observability
"""
