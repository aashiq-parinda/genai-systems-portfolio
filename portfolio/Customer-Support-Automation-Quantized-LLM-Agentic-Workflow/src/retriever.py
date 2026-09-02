"""
Compatibility wrapper for retriever module
"""
from src.core.retriever import (
    HybridPolicyRetriever,
    Document,
    ENTERPRISE_KNOWLEDGE_BASE
)

# Alias for backwards compatibility
PolicyRetriever = HybridPolicyRetriever

__all__ = [
    "HybridPolicyRetriever",
    "PolicyRetriever",
    "Document",
    "ENTERPRISE_KNOWLEDGE_BASE"
]
