# CloudFlow Customer Support RAG Observability & Evaluation Suite

A production-grade demonstration of full-stack LLM observability and automated evaluation for a customer-support RAG chatbot. This suite instruments an end-to-end RAG pipeline with **OpenTelemetry GenAI Semantic Conventions** and connects trace telemetry across three complementary observability sinks:
1. **Langfuse** (self-hosted / cloud): Full trace waterfall, prompt template versioning, and custom metric evaluation ingestion.
2. **Arize Phoenix**: RAG-specific retrieval quality, vector embedding drift, and context relevance inspection.
3. **Helicone**: High-speed edge proxy layer for token counts, real-time cost calculation, TTFT latency, and status code tracking.

The system also includes an **LLM-as-a-judge evaluation suite** that evaluates response groundedness and context relevance, injects **three deliberate failure scenarios**, and produces a cross-tool comparison matrix detailing what each platform caught, missed, and visualized.

---

## Architecture Overview

```
User Query ──► [Helicone Edge Proxy] (Tokens, Cost, Latency, Status Codes)
                     │
                     ▼
           [OpenTelemetry GenAI Tracer] (Standardized Spans)
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   [Retriever]  [Tool Call]   [LLM Synthesis]
   (SaaS Docs)  (check_order) (Prompt Version v1.2)
         │           │           │
         └───────────┼───────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    [Langfuse Sink]        [Arize Phoenix Sink]
    - Prompt Versions      - RAG Triad Metrics
    - Trace Waterfall      - Document Chunk Score
    - Custom Metric Scores - Embedding Drift
                     ▲
                     │
         [LLM-as-a-Judge Eval]
         - Groundedness (Faithfulness)
         - Context Relevance
```

---

## Project Structure

```
Fundamentals_Miscellaneous_Learning/RAG_Observability_Eval_Suite/
├── README.md                           # Architecture and operational documentation
├── requirements.txt                    # Dependencies
├── .env.example                        # Configuration template
├── docker-compose.yml                  # Self-hosted Langfuse + Arize Phoenix stack
├── run_demo.py                         # Single-command demo runner
├── data/
│   └── saas_docs/                      # Fictional CloudFlow SaaS documentation
│       ├── billing_and_plans.md        # Subscription tiers, pricing, overage fees
│       ├── sso_and_security.md         # SAML 2.0, Okta, ACS URLs, session policies
│       ├── api_rate_limits.md          # RPM limits, retry headers, 429 backoff
│       └── service_level_agreements.md # Uptime guarantees, credit schedule
├── src/
│   ├── config.py                       # Environment and path settings
│   ├── vector_store.py                 # Markdown chunking and vector retrieval
│   ├── tools.py                        # Mock `check_order_status` tool with fault injection
│   ├── rag_engine.py                   # Orchestrated RAG execution pipeline
│   ├── failure_injection.py            # The 3 deliberate failure mode definitions
│   ├── telemetry/
│   │   ├── otel_tracer.py              # OpenTelemetry GenAI semantic conventions tracer
│   │   ├── langfuse_client.py          # Langfuse prompt registry & score ingestion
│   │   ├── phoenix_client.py           # Arize Phoenix RAG openinference client
│   │   └── helicone_client.py          # Helicone edge proxy logger & cost calculator
│   └── evaluation/
│       └── llm_judge.py                # LLM-as-a-judge for groundedness & relevance
├── dashboard/
│   ├── comparison_report.py            # Comparative benchmark generator
│   └── COMPARISON_DASHBOARD.md         # Generated diagnostic analysis report
└── tests/
    └── test_rag_pipeline.py            # Automated pytest verification suite
```

---

## OpenTelemetry GenAI Semantic Conventions

Spans emitted by `src/telemetry/otel_tracer.py` adhere strictly to the OpenTelemetry Semantic Conventions for Generative AI systems:

| Attribute | Type | Description | Example |
|---|---|---|---|
| `gen_ai.system` | string | Target model provider | `"openai"` |
| `gen_ai.request.model` | string | Requested model name | `"gpt-4o-mini"` |
| `gen_ai.response.model` | string | Actual model returned | `"gpt-4o-mini"` |
| `gen_ai.usage.prompt_tokens` | int | Number of input prompt tokens | `184` |
| `gen_ai.usage.completion_tokens` | int | Number of output tokens generated | `92` |
| `gen_ai.usage.total_tokens` | int | Sum of prompt and completion tokens | `276` |
| `gen_ai.operation.name` | string | High-level pipeline stage | `"retrieval"`, `"chat"`, `"tool_execution"` |
| `gen_ai.retrieval.query` | string | User query passed into retriever | `"How to configure Okta SSO?"` |
| `gen_ai.retrieval.top_k` | int | Requested top K chunks | `3` |
| `gen_ai.retrieval.top_score` | float | Highest cosine similarity score | `0.4215` |
| `gen_ai.tool.name` | string | Name of tool invoked | `"check_order_status"` |
| `gen_ai.tool.status` | string | Execution outcome | `"SUCCESS"` or `"ERROR"` |

---

## The 3 Deliberate Failure Scenarios

### 1. Bad Retrieval (Out-of-Domain Chunks)
- **User Query**: *"How do I configure SAML 2.0 Identity Provider with Okta and what is the ACS URL?"*
- **Injected Fault**: Vector search returns marketing cookie tracking terms rather than `sso_and_security.md`.
- **Result**: Context Relevance score plummets to **0.10** (Failed).

### 2. Hallucinated LLM Answer
- **User Query**: *"Does CloudFlow have hardware acceleration or special hosting options?"*
- **Injected Fault**: Generation synthesizes ungrounded claims about *"Quantum-Encrypted On-Premise Hyper-Clusters with free lifetime hardware upgrades"*.
- **Result**: Groundedness score drops to **0.15** (Failed).

### 3. Failed Tool Call (Downstream Microservice Timeout)
- **User Query**: *"Please check the current payment status and seats for order ORD-10492."*
- **Injected Fault**: Mock `check_order_status` raises an `OrderServiceTimeoutError` (504 Gateway Timeout).
- **Result**: Tool span marked `ERROR`, Helicone logs 504 status code, tool success metric logged as **0.0**.

---

## Tool Comparison Summary

| Feature / Capability | Langfuse | Arize Phoenix | Helicone |
|---|---|---|---|
| **Primary Focus** | Tracing, prompt versioning & eval metrics | RAG retrieval quality & embedding drift | Edge proxy, token costs & latency |
| **Caught Bad Retrieval?** | 🟡 Partial (via custom eval score metric) | 🟢 Yes (RAG triad relevance & score drop) | 🔴 No (HTTP 200 returned, blind to context) |
| **Caught Hallucination?** | 🟢 Yes (Groundedness score: 0.15 logged) | 🟢 Yes (RAG triad groundedness failure) | 🔴 No (Model returned text normally) |
| **Caught Tool Failure?** | 🟢 Yes (Span status ERROR + stack trace) | 🟢 Yes (Red waterfall error node) | 🟢 Yes (HTTP 504 status code + latency) |
| **Cost & Token Tracking** | Inferred via span attributes | Inferred via span attributes | 🟢 Real-time edge cost calculation & caching |
| **Prompt Versioning** | 🟢 Built-in (`support_rag_system:v1.2.0`) | 🔴 Not a primary prompt management tool | 🟡 Prompt template hashing at proxy |

---

## Quickstart & Execution

### 1. Run Automated Tests
```bash
source /Volumes/Exty/CrackingTheGenAI/.venv/bin/activate
pytest Fundamentals_Miscellaneous_Learning/RAG_Observability_Eval_Suite/tests/ -v
```

### 2. Run the Benchmark Demo
```bash
python Fundamentals_Miscellaneous_Learning/RAG_Observability_Eval_Suite/run_demo.py
```

### 3. View the Generated Markdown Dashboard
Open `Fundamentals_Miscellaneous_Learning/RAG_Observability_Eval_Suite/dashboard/COMPARISON_DASHBOARD.md`.

### 4. Optional: Spin Up Local Self-Hosted Phoenix & Langfuse
```bash
docker compose -f Fundamentals_Miscellaneous_Learning/RAG_Observability_Eval_Suite/docker-compose.yml up -d
# Langfuse UI: http://localhost:3000
# Phoenix UI:  http://localhost:6006
```
