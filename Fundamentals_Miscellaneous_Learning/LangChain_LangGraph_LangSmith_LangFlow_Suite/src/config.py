"""
Configuration and Environment Settings for the LangChain, LangGraph, LangSmith, and LangFlow Suite.
Handles API keys, LangSmith telemetry flags, and mock vs. live LLM switching.
"""

import os
from pathlib import Path
from typing import Optional

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception as e:
        print(f"[Config] Note: Failed to read .env file: {e}")

class Config:
    # Model Provider Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # LangSmith Tracing & Observability
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "lang-ecosystem-fundamentals")

    # LangFlow Settings
    LANGFLOW_HOST: str = os.getenv("LANGFLOW_HOST", "127.0.0.1")
    LANGFLOW_PORT: int = int(os.getenv("LANGFLOW_PORT", "7860"))

    # Execution Mode:
    # If USE_MOCK_LLM is True, runs high-fidelity local simulation (0 API keys or costs required)
    USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")

    @classmethod
    def setup_langsmith_env(cls):
        """Export LangSmith environment variables to process environment."""
        os.environ["LANGCHAIN_TRACING_V2"] = cls.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = cls.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
        if cls.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY

    @classmethod
    def summary(cls) -> dict:
        return {
            "mode": "Simulated/Mock Engine (Zero Cost)" if cls.USE_MOCK_LLM else "Live API Provider",
            "openai_key_present": bool(cls.OPENAI_API_KEY),
            "anthropic_key_present": bool(cls.ANTHROPIC_API_KEY),
            "langsmith_tracing": cls.LANGCHAIN_TRACING_V2,
            "langsmith_project": cls.LANGCHAIN_PROJECT,
            "langsmith_key_present": bool(cls.LANGCHAIN_API_KEY),
            "langflow_endpoint": f"http://{cls.LANGFLOW_HOST}:{cls.LANGFLOW_PORT}",
        }
