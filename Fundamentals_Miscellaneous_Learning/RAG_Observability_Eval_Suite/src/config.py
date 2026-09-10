import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Settings:
    """Application configuration and environment settings."""
    # Run mode
    simulation_mode: bool = field(
        default_factory=lambda: os.getenv("SIMULATION_MODE", "true").lower() in ("true", "1", "yes")
    )
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    docs_dir: Path = field(init=False)
    
    # Model Provider Settings
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-mock-simulation-key")
    )
    openai_model_name: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    )
    
    # Langfuse Settings
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-mock-cloudflow")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-mock-cloudflow")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )
    
    # Arize Phoenix Settings
    phoenix_collector_endpoint: str = field(
        default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    )
    phoenix_project_name: str = field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "cloudflow-customer-support-rag")
    )
    
    # Helicone Settings
    helicone_api_key: str = field(
        default_factory=lambda: os.getenv("HELICONE_API_KEY", "sk-helicone-mock-cloudflow")
    )
    helicone_base_url: str = field(
        default_factory=lambda: os.getenv("HELICONE_BASE_URL", "https://oai.helicone.ai/v1")
    )
    
    def __post_init__(self):
        self.docs_dir = self.base_dir / "data" / "saas_docs"

# Global settings singleton
settings = Settings()
