"""Centralised configuration via pydantic-settings.

All configuration is read from environment variables (or .env file).
Nothing is hardcoded. Add fields here — never scatter os.getenv() calls
throughout the codebase.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gateway ────────────────────────────────────────────────────────────────
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8001
    gateway_env: Literal["development", "staging", "production"] = "development"
    gateway_debug: bool = False

    # ── Inference Backend ──────────────────────────────────────────────────────
    # "mock" runs a local fake SSE server. "vllm" talks to a real vLLM node.
    # This is the single toggle that lets developers without GPUs run everything.
    inference_backend: Literal["mock", "vllm"] = "mock"
    vllm_base_url: str = "http://localhost:8002"
    inference_timeout_seconds: int = 120
    inference_max_retries: int = 2

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./gateway_dev.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 20

    # ── Security ───────────────────────────────────────────────────────────────
    secret_key: str = "dev-only-insecure-key-replace-in-production"
    api_key_prefix: str = "llm_"
    api_key_hash_iterations: int = 260_000

    # ── Default Plan Limits ────────────────────────────────────────────────────
    # These are the fallback limits. Per-user overrides live in the DB.
    default_rpm: int = 10
    default_tpm: int = 50_000
    default_daily_tokens: int = 50_000
    default_monthly_tokens: int = 1_000_000
    default_max_concurrent: int = 2

    # ── CORS ───────────────────────────────────────────────────────────────────
    cors_allowed_origins: str = "http://localhost:3000"

    # ── Observability ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    enable_metrics: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.gateway_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton Settings instance. Cached after first call."""
    return Settings()
