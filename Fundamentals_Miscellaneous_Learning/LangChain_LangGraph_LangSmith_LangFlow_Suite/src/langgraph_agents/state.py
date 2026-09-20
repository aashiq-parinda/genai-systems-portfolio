"""
LangGraph State Schema Definition.
Demonstrates:
1. TypedDict State with Reducer Annotations (`Annotated[list, operator.add]`)
2. Multi-actor intermediate representation (Plan, Research, Draft, Evaluation, Approval)
3. Thread metadata and persistence keys
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

def add_messages(existing: List[Dict[str, Any]], new_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Custom reducer combining message lists (equivalent to LangGraph's add_messages)."""
    return existing + new_messages

def append_notes(existing: List[str], new_notes: List[str]) -> List[str]:
    """Append-only reducer for research findings across cycles."""
    return existing + new_notes

class AgentState(TypedDict):
    """
    Central shared state passed between nodes in the LangGraph cyclical state machine.
    """
    query: str
    thread_id: str
    plan: List[str]
    # Reducer fields: values are accumulated across cycles rather than overwritten
    research_notes: Annotated[List[str], append_notes]
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # Overwritten fields: updated to the latest value per node execution
    draft: str
    critique: str
    quality_score: float
    revision_count: int
    human_approved: Optional[bool]
    status: str
