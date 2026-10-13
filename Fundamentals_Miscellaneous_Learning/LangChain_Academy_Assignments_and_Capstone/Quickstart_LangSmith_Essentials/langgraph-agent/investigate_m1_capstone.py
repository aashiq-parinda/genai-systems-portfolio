#!/usr/bin/env python3
"""
Assignment 1 Capstone: Agent Observability Inspector
Extracts and compares threads from traces-3.json to demonstrate:
1. What the customer asked.
2. What `search_company_info` retrieved (ranking #1 vs #2).
3. What the agent told the customer.
4. Where `add_pizza_to_order` refused the order (`crust_size_unavailable`).
"""

import json
import os
from pathlib import Path

TRACES_FILE = Path(__file__).parent / "traces" / "traces-3.json"

def inspect_traces():
    if not TRACES_FILE.exists():
        print(f"Error: {TRACES_FILE} not found!")
        return

    with open(TRACES_FILE, "r") as f:
        traces = json.load(f)

    print(f"Loaded {len(traces)} traces from traces-3.json\n")
    
    # Find interesting threads
    threads_with_gluten_qa = []
    threads_with_refusals = []

    for t in traces:
        # Check inputs and outputs
        dump = json.dumps(t)
        if "crust_size_unavailable" in dump:
            threads_with_refusals.append(t)
        if "gluten" in dump.lower():
            threads_with_gluten_qa.append(t)

    print("=" * 70)
    print("🔍 STEP 1: INSPECTING A THREAD WHERE CUSTOMER ASKS ABOUT GLUTEN SIZES")
    print("=" * 70)
    
    # Inspect a trace with search_company_info query about gluten
    sample_qa = None
    for t in threads_with_gluten_qa:
        inputs = t.get("inputs", {})
        outputs = t.get("outputs", {})
        messages = inputs.get("messages", []) + outputs.get("messages", [])
        has_search = any(
            (m.get("name") or "").endswith("search_company_info") 
            for m in messages if isinstance(m, dict)
        )
        if has_search:
            sample_qa = t
            break

    if sample_qa:
        print_thread_summary(sample_qa)

    print("\n" + "=" * 70)
    print("🚨 STEP 2: INSPECTING A THREAD WHERE KITCHEN REFUSED AN ORDER")
    print("=" * 70)
    if threads_with_refusals:
        print_thread_summary(threads_with_refusals[0])

def print_thread_summary(trace):
    inputs = trace.get("inputs", {})
    outputs = trace.get("outputs", {})
    messages = []
    
    # Check top-level messages or nested messages
    if "messages" in inputs:
        messages.extend(inputs["messages"])
    if "messages" in outputs:
        messages.extend(outputs["messages"])
        
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("type")
        name = m.get("name", "")
        content = m.get("content", "")

        if role in ("user", "human"):
            print(f"\n👤 [CUSTOMER]: {content}")
        elif role in ("ai", "assistant"):
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {}).get("name") or tc.get("name")
                    args = tc.get("function", {}).get("arguments") or tc.get("args")
                    print(f"🤖 [AGENT CALLS TOOL]: {fn}({args})")
            if content:
                print(f"🤖 [AGENT SAYS]: {content}")
        elif role == "tool":
            content_preview = str(content)
            if len(content_preview) > 250:
                content_preview = content_preview[:250] + "... [truncated]"
            print(f"🔧 [TOOL RESULT ({name})]: {content_preview}")

if __name__ == "__main__":
    inspect_traces()
