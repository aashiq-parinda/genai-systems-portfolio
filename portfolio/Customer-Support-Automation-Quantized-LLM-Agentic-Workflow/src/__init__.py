"""
Customer Support Automation Package
"""
from src.core.quantization import QuantizationConfig, VRAMEstimator, QuantizationType
from src.core.classifier import TicketClassifier, RiskLevel, TicketCategory
from src.core.retriever import HybridPolicyRetriever, Document
from src.core.agent import SupportAgentWorkflow, WorkflowState

__version__ = "2.0.0"

__all__ = [
    "QuantizationConfig",
    "VRAMEstimator",
    "QuantizationType",
    "TicketClassifier",
    "RiskLevel",
    "TicketCategory",
    "HybridPolicyRetriever",
    "Document",
    "SupportAgentWorkflow",
    "WorkflowState",
]
