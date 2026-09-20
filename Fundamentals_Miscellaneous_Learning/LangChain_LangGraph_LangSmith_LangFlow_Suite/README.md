# LangChain, LangGraph, LangSmith & LangFlow Deep-Dive Suite

A production-grade fundamental learning and experimentation suite designed to master the **4 pillars of the LangChain ecosystem**:
1. **LangChain**: Modern LCEL (LangChain Expression Language), Runnables, structured Pydantic parsing, parallel branches, and fallback chains.
2. **LangGraph**: Stateful multi-actor agent orchestration, cyclic loops, self-correction, human-in-the-loop approval gates, and checkpoint persistence.
3. **LangSmith**: Production tracing, run trees, span waterfalls, latency tracking, and dataset evaluation benchmarks.
4. **LangFlow**: Official visual low-code canvas, drag-and-drop workflow execution, and pre-packaged JSON flow templates.

---

## Architecture & Ecosystem Map

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 USER / OPERATOR                         │
                  └────────────┬───────────────────────────────┬────────────┘
                               │                               │
                               ▼                               ▼
               ┌───────────────────────────────┐  ┌─────────────────────────┐
               │    LangFlow Visual Canvas     │  │   LangSmith Dashboard   │
               │   (http://127.0.0.1:7860)     │  │ (smith.langchain.com)   │
               │  - Drag-and-drop components   │  │  - Span waterfall trees │
               │  - Pre-built RAG & Agent flows│  │  - Latency & token logs │
               │  - Visual node parameterization│ │  - Benchmark evaluations│
               └───────────────┬───────────────┘  └────────────▲────────────┘
                               │                               │
                               ▼                               │
      ┌────────────────────────────────────────────────────────┴────────────────────┐
      │                        CORE EXECUTION RUNTIME                               │
      ├─────────────────────────────────────────┬───────────────────────────────────┤
      │            LangChain (LCEL)             │          LangGraph Engine         │
      │  - Unix pipe syntax (A | B | C)         │  - StateGraph & Cyclic Loops      │
      │  - RunnableParallel multi-aspect fan-out│  - Annotated[list, operator.add]  │
      │  - with_fallbacks() resilient failover  │  - interrupt_before approval gates│
      │  - Message sliding window trimmers      │  - Checkpoint state persistence   │
      └─────────────────────────────────────────┴───────────────────────────────────┘
```

---

## Project Structure

```
Fundamentals_Miscellaneous_Learning/LangChain_LangGraph_LangSmith_LangFlow_Suite/
├── README.md                            # Comprehensive architectural and execution guide
├── LANG_ECOSYSTEM_CHEATSHEET.md        # Technical syntax, reducers, and comparison matrix
├── requirements.txt                     # Curated dependencies
├── .env.example                         # Environment configuration template
├── launch_langflow.py                   # Launcher and validator for official LangFlow dashboard
├── run_langchain_demo.py                # Standalone demo: LCEL pipelines, parallel chains, fallbacks
├── run_langgraph_demo.py                # Standalone demo: Cyclic state machine with Human-in-the-Loop
├── run_langsmith_demo.py                # Standalone demo: Trace waterfalls & dataset benchmark evals
├── run_all_demos.py                     # Master verification runner testing all 4 modules
├── flows/                               # Ready-to-import flows for LangFlow visual canvas
│   ├── 01_lcel_conversational_rag.json  # RAG Pipeline (Input -> Vector DB -> Prompt -> LLM -> Output)
│   └── 02_langgraph_cyclic_agent.json   # Cyclic Agent (Planner -> Tool -> Critic -> Gate)
└── src/
    ├── config.py                        # Central configuration and LangSmith telemetry flags
    ├── mock_llm.py                      # High-fidelity zero-cost simulated LLM engine
    ├── langchain_core/
    │   ├── lcel_pipeline.py             # Pipe composition, RunnableParallel, fallback recovery
    │   └── memory_and_history.py        # Buffer windows, message trimming, rolling summarizer
    ├── langgraph_agents/
    │   ├── state.py                     # AgentState TypedDict with Annotated append reducers
    │   ├── workflow.py                  # StateGraph definition with conditional routing loops
    │   └── checkpointing.py             # MemorySaver checkpointer for state audit and replay
    ├── langsmith_observability/
    │   ├── tracing.py                   # @traceable decorator and ASCII waterfall renderer
    │   └── evaluation.py                # Dataset test cases, evaluators, and summary scorecards
    └── langflow_flows/
        ├── flow_exporter.py             # Schema validator and programmatic flow builder
        └── custom_nodes.py              # Production custom component blueprints for LangFlow
```

---

## Quickstart Guide

### 1. Verification Runner (All Modules)
Run all 4 modules end-to-end to verify functionality in seconds:
```bash
python run_all_demos.py
```

### 2. LangChain LCEL Hands-On Demo
Demonstrates Unix piping (`prompt | model | parser`), multi-aspect parallel analysis, and resilient model fallback:
```bash
python run_langchain_demo.py
```

### 3. LangGraph Cyclic Multi-Actor Agent Demo
Demonstrates stateful graph transitions, self-correcting loops (revising draft when score < 0.80), checkpointing, and human-in-the-loop interruption:
```bash
python run_langgraph_demo.py
```

### 4. LangSmith Observability & Evaluation Demo
Demonstrates nested `@traceable` span recording, ASCII waterfall visualization, and running automated dataset benchmark evaluations:
```bash
python run_langsmith_demo.py
```

### 5. Official LangFlow Visual Dashboard
Validate visual flow templates and launch the official LangFlow server:
```bash
# Validate flows and inspect environment
python launch_langflow.py --check-only

# Launch official LangFlow web UI at http://127.0.0.1:7860
python launch_langflow.py
```

---

## The 4 Pillars Explained

### 1. LangChain (LCEL)
- **Mental Model**: Linear pipeline / Directed Acyclic Graph (DAG).
- **Core Abstraction**: `Runnable` with pipe operator `|`.
- **Strengths**: Deterministic flow, type-safe streaming, structured JSON extraction, and seamless model failover using `.with_fallbacks()`.

### 2. LangGraph
- **Mental Model**: Cyclical state machine.
- **Core Abstraction**: `StateGraph`, `AgentState`, and conditional edges (`routers`).
- **Strengths**: True agentic autonomy. When an LLM generates a suboptimal output, the graph can route back to the researcher or planner rather than failing immediately. Includes native Human-in-the-Loop (`interrupt_before`) and checkpoint persistence.

### 3. LangSmith
- **Mental Model**: Production GenAI observability sink and evaluation harness.
- **Core Abstraction**: Run trees, spans, tags, latency tracking, datasets, and evaluators.
- **Strengths**: Deep visibility into latency bottlenecks, token consumption, and regression testing across model versions.

### 4. LangFlow
- **Mental Model**: Visual, low-code interface for LangChain and LangGraph.
- **Core Abstraction**: Visual nodes, typed handles, and exportable JSON flow schemas.
- **Strengths**: Rapid prototyping, visual debugging of RAG pipelines, and interactive canvas experimentation.

---

## Pre-Packaged LangFlow Visual Flows

This suite includes two production-grade flow templates located in `flows/`:

1. **`01_lcel_conversational_rag.json`**:
   - Components: User Query Input -> Qdrant Vector Store -> RAG Prompt Template -> LLM Synthesis (gpt-4o) -> Chat Output.
   - Purpose: Demonstrates vector lookup, context injection, and structured answer synthesis.

2. **`02_langgraph_cyclic_agent.json`**:
   - Components: Task Input -> Planner Node -> Tool Executor Node -> Critic Node (with revision loop back to Tool Executor) -> Human Approval Gate.
   - Purpose: Demonstrates visual agent loops and human sign-off gates.

**To Import Into LangFlow**:
1. Open LangFlow at `http://127.0.0.1:7860`
2. Click **New Flow** -> **Import**
3. Select any `.json` file from the `flows/` folder.

---

## Live API Keys vs. Zero-Cost Mock Engine

By default, `src/config.py` runs in **High-Fidelity Simulated Mode** (`USE_MOCK_LLM=true`). This provides:
- Instant execution with zero API keys required.
- Deterministic token tracking, latency simulation, and realistic agent synthesis.
- Zero billing costs during learning and experimentation.

To connect live cloud providers, simply edit `.env`:
```bash
# Set to false to invoke live LLMs
USE_MOCK_LLM=false

# Add your credentials
OPENAI_API_KEY="sk-..."
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_TRACING_V2=true
```
