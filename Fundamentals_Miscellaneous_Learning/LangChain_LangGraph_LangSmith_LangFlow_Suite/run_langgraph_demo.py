#!/usr/bin/env python3
"""
LangGraph Cyclic Multi-Actor Agent & Checkpointing Hands-On Demo.
Demonstrates:
1. Cyclic StateGraph: Planner -> Researcher -> Synthesizer -> Evaluator -> Approval Gate
2. Cyclic Edge with Self-Correction (Revision loop when score < 0.80)
3. Checkpoint Persistence: Recording state snapshots at each node transition
4. Human-in-the-Loop Interruption: Halting before gate, inspecting state, and human approval
"""

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.langgraph_agents.state import AgentState
from src.langgraph_agents.workflow import build_vector_db_research_graph
from src.langgraph_agents.checkpointing import Checkpointer

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

def run_agent_workflow():
    print_header("1. Initializing LangGraph Cyclic State Graph")
    workflow = build_vector_db_research_graph()
    checkpointer = Checkpointer()

    thread_id = "agent_thread_001"
    initial_state: AgentState = {
        "query": "Evaluate Qdrant, Pinecone, and Milvus for enterprise hybrid vector search.",
        "thread_id": thread_id,
        "plan": [],
        "research_notes": [],
        "messages": [],
        "draft": "",
        "critique": "",
        "quality_score": 0.0,
        "revision_count": 0,
        "human_approved": False,
        "status": "initialized"
    }

    current_node = workflow.entry_point
    current_state = initial_state
    checkpointer.save(thread_id, "START", current_state)

    print(f"Target Query: \"{initial_state['query']}\"")
    print(f"Thread ID:    {thread_id}")
    print(f"Entry Point:  {current_node}")

    print_header("2. Stepping through Cyclic Agent Graph")
    step_num = 1
    while current_node:
        # Check for Human-in-the-Loop Interruption Gate
        if current_node in workflow.interrupt_before and not current_state.get("human_approved", False):
            print(f"\n⏸️ [INTERRUPT TRIGGERED] Graph halted before entering: '{current_node}'")
            print("   -> Human sign-off required to finalize release.")
            break

        node_name, updates, current_state, next_node = workflow.stream_step(current_node, current_state)
        chk_id = checkpointer.save(thread_id, node_name, current_state)

        print(f"\n[Step {step_num}] Node executed: '{node_name}' (Checkpoint: {chk_id})")
        if "plan" in updates:
            print(f"   -> Plan steps generated: {len(updates['plan'])}")
        if "research_notes" in updates:
            print(f"   -> Appended research notes: {updates['research_notes'][-1][:70]}...")
        if "draft" in updates:
            print(f"   -> Synthesis draft updated ({len(updates['draft'])} chars)")
        if "quality_score" in updates:
            print(f"   -> Critic Quality Score: {updates['quality_score']:.2f} (Critique: {updates['critique']})")
            if next_node == "researcher":
                print(f"   🔄 [CYCLIC LOOP]: Score below 0.80 -> Routing back to 'researcher' for revision!")

        current_node = next_node
        step_num += 1
        time.sleep(0.05)

    print_header("3. Inspecting Checkpoint History (Time-Travel / Audit)")
    history = checkpointer.get_history(thread_id)
    print(f"Total checkpoints captured for thread '{thread_id}': {len(history)}")
    for h in history:
        print(f" - [{h['checkpoint_id']}] Node: {h['node']:<16} Status: {h['status']:<15} Rev: {h.get('revision_count')}")

    print_header("4. Human-in-the-Loop Intervention & Resumption")
    print("Human Operator Reviews Draft:")
    latest_state = checkpointer.get_latest_state(thread_id)
    print(f"\n--- Draft Preview ---\n{latest_state.get('draft')}\n---------------------")
    
    print("\nSimulating Human Action: [APPROVE DRAFT]...")
    # Human updates state to approved
    current_state = checkpointer.update_state(thread_id, {"human_approved": True})
    print("✅ Checkpoint updated: human_approved = True")

    print("\nResuming interrupted workflow through gate...")
    current_node = "approval_gate"
    node_name, updates, final_state, next_node = workflow.stream_step(current_node, current_state)
    checkpointer.save(thread_id, node_name, final_state)

    print(f"[Final Step] Node executed: '{node_name}'")
    print(f"Final Thread Status: '{final_state['status']}'")
    print(f"Quality Score:       {final_state['quality_score']:.2f}")
    print(f"Total Messages Log:  {len(final_state['messages'])} entries")

if __name__ == "__main__":
    print("\n🚀 Starting LangGraph Multi-Actor Cyclic Agent Demonstration...")
    run_agent_workflow()
    print("\n✅ LangGraph Cyclic Agent Workflow Completed Successfully!\n")
