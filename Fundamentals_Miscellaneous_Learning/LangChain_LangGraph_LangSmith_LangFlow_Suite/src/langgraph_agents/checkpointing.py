"""
LangGraph Checkpointing & Human-in-the-Loop Persistence.
Demonstrates:
1. Thread-isolated checkpoints (MemorySaver / SQLite style)
2. State Snapshot History & Time-Travel inspection
3. State editing before resuming an interrupted thread
"""

import copy
import time
from typing import Dict, Any, List, Optional
from .state import AgentState

class CheckpointRecord:
    def __init__(self, thread_id: str, checkpoint_id: str, node: str, state: AgentState):
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id
        self.node = node
        self.state = copy.deepcopy(state)
        self.timestamp = time.time()

class Checkpointer:
    """
    In-memory / persistence engine mirroring LangGraph's MemorySaver.
    Tracks state snapshots at each step of graph execution.
    """
    def __init__(self):
        # thread_id -> list of CheckpointRecord
        self.storage: Dict[str, List[CheckpointRecord]] = {}

    def save(self, thread_id: str, node: str, state: AgentState) -> str:
        if thread_id not in self.storage:
            self.storage[thread_id] = []
        
        checkpoint_id = f"chk_{len(self.storage[thread_id]) + 1}_{int(time.time() * 1000)}"
        record = CheckpointRecord(thread_id, checkpoint_id, node, state)
        self.storage[thread_id].append(record)
        return checkpoint_id

    def get_latest_state(self, thread_id: str) -> Optional[AgentState]:
        records = self.storage.get(thread_id)
        if not records:
            return None
        return copy.deepcopy(records[-1].state)

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        records = self.storage.get(thread_id, [])
        return [
            {
                "checkpoint_id": r.checkpoint_id,
                "node": r.node,
                "status": r.state.get("status"),
                "revision_count": r.state.get("revision_count"),
                "timestamp": r.timestamp
            }
            for r in records
        ]

    def update_state(self, thread_id: str, updates: Dict[str, Any]) -> AgentState:
        """Allows human / supervisor to edit state before resuming execution."""
        latest = self.get_latest_state(thread_id)
        if not latest:
            raise KeyError(f"Thread ID {thread_id} not found in checkpointer.")
        
        latest.update(updates)
        self.save(thread_id, "human_override", latest)
        return latest
