# CloudFlow RAG Observability & Evaluation Comparison Report

## Executive Summary
This report analyzes how three leading GenAI observability platforms (**Langfuse**, **Arize Phoenix**, and **Helicone**) behave when monitoring a production-style RAG pipeline instrumented with **OpenTelemetry GenAI Semantic Conventions**.

We evaluated a healthy baseline pipeline and three injected failure modes:
1. **Bad Retrieval**: Vector store fetched out-of-domain cookie tracking policy instead of SAML SSO guide.
2. **Hallucination**: LLM fabricated ungrounded "Quantum-Encrypted Hyper-Clusters".
3. **Failed Tool Call**: Downstream `check_order_status` microservice suffered a 504 gateway timeout.

---

## Cross-Tool Diagnostic Matrix

+----------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| Scenario                               | Langfuse                                                                     | Arize Phoenix                                                                                                                         | Helicone                                                                                                    |
|                                        | (Tracing + Prompts + Metrics)                                                | (RAG Retrieval & Drift)                                                                                                               | (Edge Proxy & Costs)                                                                                        |
+========================================+==============================================================================+=======================================================================================================================================+=============================================================================================================+
| Baseline (Healthy)                     | 🟢 CAUGHT:                                                                   | 🟢 CAUGHT:                                                                                                                            | 🟢 CAUGHT:                                                                                                  |
|                                        | Traces all spans                                                             | Retrieval similarity: 0.42                                                                                                            | Prompt: 184 tok                                                                                             |
|                                        | Records Groundedness: 0.85                                                   | Chunk score distribution                                                                                                              | Comp: 92 tok                                                                                                |
|                                        | Records Relevance: 0.88                                                      | Span duration tree                                                                                                                    | Cost: $0.000083                                                                                             |
|                                        | Logs prompt version v1.2                                                     |                                                                                                                                       | Status: 200 OK                                                                                              |
+----------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| Failure 1: Bad Retrieval               | 🟡 PARTIALLY CAUGHT:                                                         | 🟢 FULLY CAUGHT & VISUALIZED:                                                                                                         | 🔴 MISSED:                                                                                                  |
| (Query: SAML SSO,                      | Shows low score (0.10) for custom metric 'context_relevance',                | Flags retrieval anomaly in RAG triad view. Document text diff shows cookie policy mismatch. Flags low cosine score in retrieval node. | Reported 200 OK, normal latency & cost. Cannot inspect retrieved document quality or vector drift.          |
| Retrieved: Cookie Policy)              | but doesn't show embedding cluster drift natively without custom code.       |                                                                                                                                       |                                                                                                             |
+----------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| Failure 2: Hallucination               | 🟢 FULLY CAUGHT:                                                             | 🟢 FULLY CAUGHT:                                                                                                                      | 🔴 MISSED:                                                                                                  |
| (LLM invents 'Quantum Hyper-Clusters') | LLM-judge groundedness metric logged at 0.15 (FAILED).                       | RAG Triad groundedness metric triggers alert. Flags discrepancy between context and output tokens.                                    | Reported 200 OK, token cost $0.000072. Edge proxy has zero awareness that the text is factually fabricated. |
|                                        | Prompt version tracked. Inspection of generation node shows fabricated text. |                                                                                                                                       |                                                                                                             |
+----------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| Failure 3: Failed Tool Call            | 🟢 FULLY CAUGHT:                                                             | 🟢 FULLY CAUGHT:                                                                                                                      | 🟢 CAUGHT AT EDGE:                                                                                          |
| (check_order_status 504 Timeout)       | Span 'tool_call:check_order_status' marked STATUS=ERROR.                     | Tool execution span marked ERROR in red waterfall.                                                                                    | Status 504 recorded.                                                                                        |
|                                        | Error event with stack trace attached.                                       | Pinpoints exact microservice timeout exception in span inspector.                                                                     | Error rate spikes on Helicone dashboard.                                                                    |
|                                        | Custom score logged as 0.0.                                                  |                                                                                                                                       | Surfaces latency spike immediately at edge proxy.                                                           |
+----------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+

---

## Detailed Tool Breakdown

### 1. Langfuse (Self-Hosted Tracing, Prompt Management & Eval Metrics)
- **Strengths**:
  - Unbeatable for end-to-end trace waterfalls and prompt version control (linked to `support_rag_system:v1.2.0`).
  - Seamlessly receives custom LLM-judge scores (`groundedness`, `context_relevance`, `tool_execution_success`) attached directly to traces.
  - Granular span-by-span exception tracking with OpenTelemetry GenAI attributes.
- **Blind Spots**:
  - Does not provide native vector embedding visualization or automatic retrieval drift analytics out-of-the-box (requires custom evaluation metric code).

### 2. Arize Phoenix (RAG Retrieval Quality & Semantic Drift)
- **Strengths**:
  - Specifically designed for RAG triad diagnostics (Context Relevance, Faithfulness/Groundedness, Question Answering).
  - Inspects document chunks directly inside the `RETRIEVER` span with relevance score distributions.
  - Instantly exposes when vector similarity drops or retrieval noise enters the context window.
- **Blind Spots**:
  - Not an API gateway; does not provide edge rate-limiting, caching, or billing proxy capabilities.

### 3. Helicone (Lightweight Proxy Layer for Costs & Latency)
- **Strengths**:
  - Zero-latency edge monitoring: captures exact token consumption (prompt vs completion), real-time dollar cost, and HTTP status codes.
  - Instantly caught the 504 gateway timeout and latency spikes.
  - Allows session and user tagging via lightweight HTTP headers without polluting pipeline code.
- **Blind Spots**:
  - Has zero visibility into the internal RAG retrieval step or LLM hallucination—if the model generates a hallucinated answer with HTTP 200, Helicone treats it as a 100% successful call.

---

## Aggregated Pipeline Metrics
- **Total Invocations Tested**: 4
- **Helicone Total Cost Recorded**: $0.000582
- **Helicone Error Rate**: 25.0%
- **Langfuse Custom Scores Logged**: 10
