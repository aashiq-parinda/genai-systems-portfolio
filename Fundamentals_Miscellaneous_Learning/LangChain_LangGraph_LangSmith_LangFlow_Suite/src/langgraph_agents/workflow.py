"""
LangGraph Agentic Workflow: Cyclic State Machine with Human-in-the-Loop.
Demonstrates:
1. Node Functions (Planner -> Researcher -> Synthesizer -> Evaluator -> Approval Gate)
2. Cyclic Conditional Edges (Self-Correction / Re-query Loop)
3. Human-in-the-Loop Approval Interruption (interrupt_before)
4. State transitions and state-delta inspection
"""

import time
from typing import Dict, Any, List, Optional, Callable
from .state import AgentState
from ..mock_llm import MockLLM

# ------------------------------------------------------------------------------
# Node Implementation Functions
# ------------------------------------------------------------------------------

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Breaks down user query into actionable sub-tasks."""
    llm = MockLLM(model_name="mock-planner-gpt4o", latency_ms=60)
    prompt = f"Break down query into plan subtasks: {state['query']}"
    resp = llm.invoke(prompt)
    
    plan_steps = [
        "Step 1: Benchmark retrieval latency & memory footprints for vector DBs",
        "Step 2: Collect real-world filtering and clustering capabilities",
        "Step 3: Structure comprehensive production trade-off analysis",
        "Step 4: Present recommendations for human approval"
    ]
    return {
        "plan": plan_steps,
        "status": "planned",
        "messages": [{"role": "planner", "content": f"Formulated {len(plan_steps)} execution steps."}]
    }

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Executes tool retrieval / data gathering for the plan."""
    revision = state.get("revision_count", 0)
    llm = MockLLM(model_name="mock-researcher-gpt4o", latency_ms=80)
    
    findings = (
        f"[Revision {revision}] Extracted facts: Qdrant delivers sub-5ms HNSW filtering with memory-mapped storage. "
        "Milvus scales to billions using segmented segments on MinIO. Pinecone serverless simplifies cold-start indexing."
    )
    return {
        "research_notes": [findings],
        "status": "researched",
        "messages": [{"role": "researcher", "content": f"Gathered technical benchmarks for revision {revision}."}]
    }

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Compiles research notes into a coherent synthesis draft."""
    notes_combined = " | ".join(state.get("research_notes", []))
    draft = (
        f"### Executive Technical Report: Vector Database Architecture\n\n"
        f"**Target Objective**: {state['query']}\n\n"
        f"**Synthesis**:\n{notes_combined}\n\n"
        f"**Recommendation**: Choose Qdrant for rich payload filtering on Kubernetes; "
        f"choose Pinecone for minimal DevOps overhead; choose Milvus for massive multi-billion clusters."
    )
    return {
        "draft": draft,
        "status": "synthesized",
        "messages": [{"role": "synthesizer", "content": "Generated structured draft synthesis."}]
    }

def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Evaluator / Critic node assessing quality against SLA threshold."""
    revision = state.get("revision_count", 0)
    # Simulate a learning curve: first revision scores 0.72 (triggering loop), second scores 0.91 (passes)
    quality_score = 0.72 if revision == 0 else 0.92
    critique = (
        "Draft lacks deeper hardware allocation details on memory-mapped indices."
        if quality_score < 0.80 else
        "Draft satisfies technical completeness and accuracy standards."
    )

    return {
        "quality_score": quality_score,
        "critique": critique,
        "revision_count": revision + 1,
        "status": "evaluated",
        "messages": [{"role": "evaluator", "content": f"Assessed draft quality score: {quality_score:.2f}."}]
    }

def approval_gate_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Human-in-the-Loop gate. Only finishes if human_approved is True."""
    approved = state.get("human_approved", False)
    return {
        "status": "approved_and_completed" if approved else "awaiting_human_approval",
        "messages": [{"role": "gatekeeper", "content": f"Human approval status: {approved}"}]
    }

# ------------------------------------------------------------------------------
# Conditional Edge Router
# ------------------------------------------------------------------------------

def router_evaluator_decision(state: AgentState) -> str:
    """
    Evaluator conditional edge:
    - If score < 0.80 and revision < 2: loop back to 'researcher' (Cyclic Graph)
    - Else: route forward to 'approval_gate'
    """
    if state["quality_score"] < 0.80 and state.get("revision_count", 0) < 2:
        return "researcher"
    return "approval_gate"

# ------------------------------------------------------------------------------
# LangGraph Workflow Orchestrator
# ------------------------------------------------------------------------------

class StateGraphEngine:
    """
    Complete stateful orchestration engine mirroring LangGraph StateGraph capabilities:
    - Directed cyclical graphs
    - Reducer-aware state updates
    - Step-by-step stream generation
    - Human-in-the-loop interruption gates
    """
    def __init__(self):
        self.nodes: Dict[str, Callable[[AgentState], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[AgentState], str]] = {}
        self.interrupt_before: List[str] = []
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, func: Callable[[AgentState], Dict[str, Any]]):
        self.nodes[name] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, start_node: str, end_node: str):
        self.edges[start_node] = end_node

    def add_conditional_edges(self, source_node: str, router_func: Callable[[AgentState], str]):
        self.conditional_edges[source_node] = router_func

    def compile(self, interrupt_before: Optional[List[str]] = None):
        if interrupt_before:
            self.interrupt_before = interrupt_before
        return self

    def _apply_updates(self, current_state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """Applies state changes respecting append reducers."""
        next_state = dict(current_state)
        for k, v in updates.items():
            if k in ("research_notes", "messages"):
                # Apply reducer logic
                existing = current_state.get(k, [])
                next_state[k] = existing + (v if isinstance(v, list) else [v])
            else:
                next_state[k] = v
        return next_state  # type: ignore

    def stream_step(self, current_node: str, state: AgentState):
        """Execute a single node and return updated state and next node."""
        node_func = self.nodes[current_node]
        updates = node_func(state)
        new_state = self._apply_updates(state, updates)

        # Determine next node
        if current_node in self.conditional_edges:
            next_node = self.conditional_edges[current_node](new_state)
        elif current_node in self.edges:
            next_node = self.edges[current_node]
        else:
            next_node = None

        return current_node, updates, new_state, next_node

def build_vector_db_research_graph() -> StateGraphEngine:
    """Builds and compiles the full cyclic LangGraph workflow."""
    workflow = StateGraphEngine()

    # 1. Register Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("approval_gate", approval_gate_node)

    # 2. Set Entry Point
    workflow.set_entry_point("planner")

    # 3. Add Edges
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "synthesizer")
    workflow.add_edge("synthesizer", "evaluator")

    # 4. Add Cyclical Conditional Edge from Evaluator
    workflow.add_conditional_edges("evaluator", router_evaluator_decision)

    # 5. Compile with Human-in-the-Loop Interruption
    workflow.compile(interrupt_before=["approval_gate"])
    return workflow
