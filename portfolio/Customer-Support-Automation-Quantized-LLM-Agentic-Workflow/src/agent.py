"""
Compatibility wrapper for agent module
"""
from src.core.agent import (
    SupportAgentWorkflow,
    WorkflowState,
    COST_PER_MILLION_TOKENS
)

__all__ = [
    "SupportAgentWorkflow",
    "WorkflowState",
    "COST_PER_MILLION_TOKENS"
]
