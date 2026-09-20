# Master Cheatsheet: LangChain vs LangGraph vs LangSmith vs LangFlow

A consolidated architectural reference and mental model guide for modern GenAI application engineering.

---

## 1. Executive Comparison Matrix

| Dimension | LangChain (LCEL) | LangGraph | LangSmith | LangFlow |
| :--- | :--- | :--- | :--- | :--- |
| **Core Paradigm** | Directed Acyclic Graph (DAG) / Linear Pipe | Cyclic, Stateful Multi-Actor State Machine | GenAI Telemetry, Tracing & Eval Platform | Visual Node-Based Low-Code Canvas |
| **Primary Abstraction** | `RunnableSequence` (`prompt \| model \| parser`) | `StateGraph`, Nodes, Edges, Reducers | `RunTree`, Spans, Datasets, Evaluators | Visual Nodes, Handles, Flow JSON |
| **Control Flow** | Linear, Sequential, Parallel fan-out | Cyclic loops, self-correction, branching | Passive observation & scheduled evals | Interactive visual canvas & execution engine |
| **State Handling** | Stateless / Thread-local Buffer | Central `AgentState` with Annotated Reducers | Immutable trace runs & session logs | Node-to-node data payloads |
| **Human-in-the-Loop** | Manual input pauses (clunky) | Native `interrupt_before` / `interrupt_after` | Annotator queues & human feedback labels | Visual user input nodes & pause gates |
| **Persistence** | Simple memory buffers | `MemorySaver`, SQLite, Postgres Checkpointers | Hosted cloud / Postgres backend | SQLite / PostgreSQL project storage |
| **Best Used For** | Retrieval chains, structured extraction, deterministic prompt flows | Autonomous agents, multi-step research, self-critiquing loops | Production tracing, latency audits, regression benchmarks | Rapid prototyping, visual debugging, low-code demos |
| **When NOT to Use** | Long-running multi-agent loops with cycles | Simple single-turn prompt-response tasks | Purely local air-gapped dev without telemetry needs | Production enterprise codebases requiring strict Git CI/CD |

---

## 2. LangChain: LCEL (LangChain Expression Language)

### The Unix Pipe Philosophy
LangChain v0.2+ is centered around the `Runnable` protocol, allowing Unix-style piping (`|`):

```python
# Canonical Pipeline: Prompt -> Model -> Structured Parser
chain = prompt_template | chat_model | pydantic_output_parser
result = chain.invoke({"query": "Explain HNSW indexing"})
```

### Key LCEL Primitives

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch

# 1. RunnableParallel: Execute multiple chains concurrently and combine outputs
analysis_chain = RunnableParallel({
    "summary": prompt_summary | llm | StrOutputParser(),
    "entities": prompt_entities | llm | JsonOutputParser(),
    "original_query": RunnablePassthrough()
})

# 2. Resilient Fallbacks (.with_fallbacks)
resilient_llm = gpt_4o.with_fallbacks([claude_3_5_sonnet, gpt_4o_mini])

# 3. Dynamic Routing (RunnableBranch)
branch = RunnableBranch(
    (lambda x: "billing" in x["intent"], billing_chain),
    (lambda x: "tech" in x["intent"], tech_support_chain),
    default_chain
)
```

---

## 3. LangGraph: Cyclic Multi-Actor State Graphs

Vanilla LangChain cannot easily handle loops (e.g. self-correction where an agent critiques its draft and retries). **LangGraph** was built specifically to model cyclical, stateful agent graphs.

### State & Reducers

In LangGraph, state is a shared schema where fields can be overwritten or accumulated via **reducers**:

```python
import operator
from typing import TypedDict, Annotated, List

class AgentState(TypedDict):
    query: str
    # 'operator.add' ensures new items are appended rather than overwriting existing ones!
    messages: Annotated[List[dict], operator.add]
    research_notes: Annotated[List[str], operator.add]
    # Overwritten fields:
    draft: str
    quality_score: float
    revision_count: int
```

### Graph Construction & Conditional Routing

```python
from langgraph.graph import StateGraph, END

# 1. Initialize with State
workflow = StateGraph(AgentState)

# 2. Register Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("approval_gate", approval_gate_node)

# 3. Define Flow Edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "synthesizer")
workflow.add_edge("synthesizer", "evaluator")

# 4. Define Conditional Edge (The Loop!)
def route_evaluation(state: AgentState) -> str:
    if state["quality_score"] < 0.80 and state["revision_count"] < 3:
        return "researcher"  # LOOP BACK!
    return "approval_gate"

workflow.add_conditional_edges("evaluator", route_evaluation)

# 5. Compile with Human-in-the-Loop Interruption
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval_gate"]  # Pauses thread before running gate
)
```

---

## 4. LangSmith: Observability & Evaluation

LangSmith provides enterprise-grade observability and dataset benchmarking for any LLM system.

### Environment Setup

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="lsv2_pt_..."
export LANGCHAIN_PROJECT="enterprise-production-agents"
```

### Automatic & Manual Tracing (`@traceable`)

```python
from langsmith import traceable

@traceable(name="hybrid_vector_search", run_type="retriever")
def search_vector_db(query: str, top_k: int = 4):
    # Traced automatically with inputs, outputs, duration, and metadata
    return qdrant_client.search(query=query, limit=top_k)

@traceable(name="agent_reasoning_step", run_type="chain")
def run_agent_turn(user_message: str):
    docs = search_vector_db(user_message)
    return llm.invoke(...)
```

### Dataset Benchmarks & Evaluators

```python
from langsmith.evaluation import evaluate

# Define custom LLM-as-a-judge or exact-match evaluator
def groundedness_evaluator(run, example):
    prediction = run.outputs.get("output")
    context = run.inputs.get("context")
    score = 1.0 if verify_facts(prediction, context) else 0.0
    return {"key": "groundedness", "score": score}

# Run benchmark against test dataset
results = evaluate(
    target_chain,
    data="enterprise-qa-gold-dataset",
    evaluators=[groundedness_evaluator],
    experiment_prefix="eval-v2.1"
)
```

---

## 5. LangFlow: Visual Canvas & Dashboard

LangFlow translates code-based LangChain and LangGraph topologies into a drag-and-drop web UI.

### Launching the Dashboard

```bash
# Start official LangFlow visual server on port 7860
langflow run --host 127.0.0.1 --port 7860

# Or using the suite launcher
python launch_langflow.py
```

### Importing Pre-Packaged Flows
1. Open `http://127.0.0.1:7860`
2. Click **New Flow** -> **Import**
3. Select any file from `flows/`:
   - `flows/01_lcel_conversational_rag.json` (Vector Store + Retriever + Prompt + LLM)
   - `flows/02_langgraph_cyclic_agent.json` (Planner + Tools + Evaluator Loop + Human Gate)

### Custom Component Pattern

```python
from langflow.custom import CustomComponent

class VectorStoreRouter(CustomComponent):
    display_name = "Vector DB Router"
    description = "Routes query between Pinecone and Qdrant"
    
    def build_config(self):
        return {
            "query": {"field_type": "str"},
            "database": {"field_type": "str", "options": ["Qdrant", "Pinecone"]}
        }
        
    def build(self, query: str, database: str = "Qdrant"):
        return {"selected_db": database, "query": query}
```
