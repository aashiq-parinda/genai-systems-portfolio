# 🏛️ The MAANG GenAI System Design Whiteboard Playbook
## 📐 Architectural Blueprints, Latency Budgets & Scalable Diagrams for the 4 Classic GenAI Systems

> 🎯 **Target Stage**: 45–60 Minute GenAI System Design & Architecture Whiteboard Rounds  
> 🏷️ **Target Roles**: Senior GenAI Engineer (L5/E5) | Staff AI Architect (L6+) | Principal Distributed Systems Lead (L7)  
> 🏢 **Target Companies**: Meta, Google (DeepMind), Amazon (AWS/Bedrock), Apple, Netflix, OpenAI, Anthropic, Databricks  
> 📌 **Companion Guides**: [🎙️ Verbal Tech Guide](GENAI_VERBAL_TECHNICAL_INTERVIEW_GUIDE.md) | [🤝 Behavioral & Leadership Guide](GENAI_BEHAVIORAL_AND_LEADERSHIP_ROUND.md) | [💻 Coding Bible](MAANG_GENAI_INTERVIEW_GUIDE.md)

---

## ⚡ The 6-Step Whiteboard Attack Blueprint

Whenever given a prompt in a GenAI system design interview, immediately execute this structured rhythm on the whiteboard:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  CLARIFY REQUIREMENTS & LATENCY SLAS (TTFT, ITL, QPS, P99)               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2️⃣  DATA MODEL & INGESTION (AST, Chunking, Embeddings, RBAC Metadata)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3️⃣  END-TO-END DATA FLOW & SERVING ARCHITECTURE (Gateway, Router, Cache)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4️⃣  GPU HARDWARE & CAPACITY SIZING (VRAM, KV Cache, Bandwidth, TP/PP)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5️⃣  GUARDRAILS, EVALS & CACHING (PII, Jailbreak, Semantic Cache, Ragas)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6️⃣  FAILURE MODES & RESILIENCY (OOM, Cold Starts, Degradation, Fallbacks)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📌 Table of Contents

1. [📐 System 1: Real-Time Code Copilot & Autocomplete Engine (Cursor / GitHub Copilot)](#1️⃣-system-1-real-time-code-copilot--autocomplete-engine)
2. [🎙️ System 2: Sub-300ms Real-Time Voice Assistant (GPT-4o / Gemini Live)](#2️⃣-system-2-sub-300ms-real-time-voice-assistant)
3. [🏢 System 3: Enterprise Context Engine with Real-Time Multi-Tenant ACL & Instant Invalidation](#3️⃣-system-3-enterprise-context-engine-with-real-time-multi-tenant-acl)
4. [📊 System 4: 100M Requests/Day GenAI Telemetry, FinOps & Continuous Eval Platform](#4️⃣-system-4-100m-requestsday-genai-telemetry-finops--continuous-eval-platform)

---

## 1️⃣ System 1: Real-Time Code Copilot & Autocomplete Engine

> **Interview Prompt**: *"Design a real-time code completion engine like GitHub Copilot or Cursor that suggests code as the developer types inside an IDE across millions of developers."*

### 1. Requirements & Latency Budget
- **Latency SLA**: Total keystroke-to-suggestion latency **P95 < 80ms, P99 < 120ms**. If latency exceeds 150ms, the developer has already typed past the suggestion point, making it useless.
- **Throughput**: 100,000 requests per second (RPS) peak globally.
- **Accuracy**: Acceptance rate > 30%, zero syntax errors.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TOTAL BUDGET: 80ms (P95)                              │
│                                                                             │
│  Debounce  ──► Network (HTTP/2) ──► Prompt Engine ──► Model TTFT ──► Stream │
│   (20ms)           (15ms)              (10ms)           (25ms)        (10ms)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. The Core Secret: Fill-in-the-Middle (FIM) Architecture
Standard causal LLMs only predict text *after* the prompt. Code completion happens *in the middle* of a file between existing code above and existing code below.

We format training and inference using **FIM Special Tokens**:
$$\text{Input Prompt} = \langle \text{PRE} \rangle \, \text{Prefix Code (Above Cursor)} \, \langle \text{SUF} \rangle \, \text{Suffix Code (Below Cursor)} \, \langle \text{MID} \rangle$$
The model then generates strictly what belongs between `<MID>` and `<EOM>`.

### 3. Architecture & Data Flow Diagram

```text
IDE / Client (VS Code / JetBrains)
   │
   ├──► [1. Debounce Gate (30ms)] ── (Skip if fast keystrokes)
   │
   ├──► [2. AST Tree-Sitter & Import Analyzer]
   │       ├── Extract enclosing function & class definitions
   │       └── Extract imported types & recently opened files
   │
   ▼
Edge Anycast POP / Gateway (Cloudflare / Envoy)
   │
   ├──► [3. Local Client & Edge Cache] (Exact prefix match hash)
   │
   ▼
Model Inference Cluster (vLLM / TensorRT-LLM)
   │
   ├── [4. Speculative Drafting Engine]
   │     ├── Draft Model: 1B StarCoder / Qwen-Coder (Runs in 10ms on Tensor Cores)
   │     └── Target Model: 7B Code-LLM (Validates in 1 parallel forward step)
   │
   ▼
Streaming Response back to IDE (via HTTP/2 SSE / WebSocket)
```

### 4. Key Architectural Trade-offs & Deep-Dive Talking Points:
1. **Debouncing Strategy**: Never fire an inference request on every raw keystroke. Debounce for 30–50ms. If the user is actively typing a burst of characters, keep the request pending.
2. **Context Window Pruning**: Don't dump the entire 5,000-line file into context. Use Tree-Sitter AST to select:
   - Current enclosing scope (function/class) up to 60 lines.
   - Suffix: next 30 lines below cursor.
   - Related context: Signatures from imported modules (BM25 search over workspace symbols).
3. **Model Sizing Decision**: Never use a 70B model for in-line autocomplete—latency will always exceed 300ms. Use a distilled 1.5B–3B parameter SLM fine-tuned exclusively on FIM datasets.

---

## 2️⃣ System 2: Sub-300ms Real-Time Voice Assistant

> **Interview Prompt**: *"Design a real-time conversational voice assistant (like GPT-4o or Gemini Live) where a user can speak naturally, hear streaming vocal responses, and interrupt the AI mid-sentence."*

### 1. The Hard Engineering Constraint: Glass-to-Glass Latency < 300ms
Human conversational turn-taking happens in **200ms to 300ms**. Anything above 400ms feels like an awkward walkie-talkie conversation.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 300ms "GLASS-TO-GLASS" LATENCY BUDGET                       │
│                                                                             │
│ VAD Detection ──► Streaming ASR ──► LLM TTFT ──► Chunked TTS ──► Audio Pack │
│    (20ms)            (70ms)           (100ms)       (80ms)         (30ms)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Architecture: Cascade vs. Native Multimodal

| Dimension | Cascaded Pipeline (VAD ➔ ASR ➔ LLM ➔ TTS) | Native Speech-to-Speech (End-to-End) |
| :--- | :--- | :--- |
| **How it Works** | Audio $\to$ Text $\to$ LLM $\to$ Text $\to$ Audio | Audio Spectrogram tokens directly into Transformer |
| **Pros** | Modular, cheap, debuggable, easy guardrails | Captures emotional tone, laughter, sighs, zero ASR error propagation |
| **Cons** | Error compounding, stripped emotion, higher latency | High GPU cost, hard to safety guardrail intermediate thoughts |
| **Recommendation** | **Standard MAANG Production Reality**: Cascaded streaming pipeline with chunked early synthesis. |

### 3. High-Concurrency Voice Pipeline Architecture

```text
User Microphone (Opus 16kHz via WebRTC / Full-Duplex WebSocket)
   │
   ▼
[1. Silero VAD (Voice Activity Detection)]
   ├── Runs on CPU at edge (<10ms per 30ms audio frame)
   ├── Detects user speech onset & endpoint
   │
   ▼
[2. Streaming ASR Engine (Whisper / Conformer Streaming)]
   ├── Emits partial transcriptions every 50ms
   ├── Speculative prompt pre-warming on partial transcript
   │
   ▼
[3. LLM Orchestrator (vLLM Streaming Engine)]
   ├── Ingests finalized sentence tokens
   ├── Emits completion tokens over SSE
   │
   ▼
[4. Sentence-Chunking Buffer]
   ├── Buffers tokens until first punctuation mark (, . ? !)
   │
   ▼
[5. Streaming TTS Engine (FastSpeech2 / StyleTTS2 / VITS)]
   ├── Synthesizes audio chunk in 50ms
   │
   ▼
Audio Player (Streams Opus packets to user speaker)
```

### 4. Handling Real-Time User Interruption (Barge-In)
When the user speaks while the AI is talking:
1. **Client VAD triggers immediately** on incoming mic audio.
2. Client sends an immediate control frame: `{"type": "INTERRUPT"}` over the WebRTC data channel.
3. Server **immediately terminates** downstream TTS synthesis and LLM token generation.
4. The server **evicts the unuttered tokens from the KV Cache**, preserving only the words the user actually heard in conversation history.

---

## 3️⃣ System 3: Enterprise Context Engine with Real-Time Multi-Tenant ACL

> **Interview Prompt**: *"Design an enterprise RAG knowledge platform for 500,000 employees connecting Google Drive, Slack, Confluence, and Jira with strict role-based access control (RBAC) and instant permission revocation."*

### 1. The Core Architectural Trap: Pre-Filtering vs. Post-Filtering

> [!CAUTION]
> **Why 90% of Candidates Fail This Round**: They propose **Post-Filtering** (Search top 50 documents, then discard documents the user cannot view).  
> **Catastrophic Failure**: If a CEO query searches financial results, and the top 50 results belong to an unreleased M&A document that an engineer lacks permission to see, post-filtering discards all 50 results, returning *"I don't know"*, even though 5 other public documents answered the query!

### 2. The Solution: Pre-Filtered Hierarchical HNSW with Tenant Bitmasks
Every document chunk is indexed alongside an **Access Control List (ACL) bitmask or tenant set**.

```text
Document Vector Ingestion:
Chunk Vector ──► [HNSW Vector Index]
                 └── Stored Metadata: [tenant_id: "acme", acl_groups: ["eng-leads", "finance-ro"]]
```

At query time, the user's JWT token is decoded into their authorized group IDs:
$$\text{User Clearances} = \{\text{group\_102}, \text{group\_405}, \text{user\_8892}\}$$
During the HNSW graph traversal, nodes are **evaluated against the bitmask during candidate neighbor expansion**. Non-permitted vectors are never visited.

### 3. Architecture & Data Flow

```text
Enterprise Connectors (Google Drive, Slack, Confluence)
   │
   ├──► [Change-Data-Capture (CDC) via Debezium / Webhooks]
   │
   ▼
Async Ingestion Pipeline (Kafka / Temporal)
   │
   ├── [1. Document Hasher & Deduplication] (Skip unchanged files)
   ├── [2. Structural Chunker] (Markdown header & table preservation)
   ├── [3. Presidio PII Masking Filter]
   ├── [4. Embedding Generation] (bge-large / text-embedding-3)
   │
   ▼
Multi-Tenant Vector DB (Qdrant / Milvus / pgvector)
   ├── Tenant Partitioning (Namespaces per organization)
   └── Pre-filtering ACL Bitmask Index
   │
   ▲
Query Pipeline
   ├── 1. Gateway validates User JWT & extracts RBAC clearance list
   ├── 2. Pre-filtered Vector Search (HNSW) + BM25 Lexical Search
   ├── 3. RRF Rank Fusion ──► Top 40 Chunks
   ├── 4. Cross-Encoder Reranker ──► Top 5 Chunks
   └── 5. LLM Synthesis with Groundedness Verification
```

### 4. Handling Instant Permission Revocation (< 1s SLA)
When an employee is removed from a Slack channel or project:
- Updating millions of chunk ACLs across vector indexes is too slow ($O(N)$ writes).
- **The O(1) Solution**: The vector index only stores an abstract `resource_id` or `channel_id`.
- The user's active permissions are cached in a Redis **User-Permission Bitset** with a 10-second TTL and immediate pub/sub invalidation on ACL changes. The vector search pre-filter evaluates against the live Redis permission set.

---

## 4️⃣ System 4: 100M Requests/Day GenAI Telemetry, FinOps & Continuous Eval Platform

> **Interview Prompt**: *"Design a global telemetry, cost attribution, and automated evaluation platform that monitors, traces, and evaluates every LLM call across an enterprise serving 100M requests/day."*

### 1. Scale & Data Math
- **100M requests/day** $\approx$ 1,160 average RPS, **Peak 4,000 RPS**.
- If each trace contains prompt, completion, latency, and metadata ($\approx 2 \text{ KB}$), total data volume is:
  $$100{,}000{,}000 \times 2 \text{ KB} = 200 \text{ GB / day} \approx 6 \text{ TB / month}$$
- **Key Challenge**: Running LLM-as-a-Judge evaluations on 100M requests would cost millions of dollars in secondary LLM calls.

### 2. Tiered Sampling & Evaluation Architecture

```text
Client Applications / Gateways
   │
   ├── [1. OpenTelemetry Tracing SDK] (Async non-blocking ring buffer)
   │     Emits: TraceID, SpanID, Model, TTFT, ITL, Tokens, Prompt, Output
   │
   ▼
Kafka / AWS Kinesis Ingestion Stream (4,000 RPS)
   │
   ├──► [Route A: 100% Real-Time Metrics Path]
   │       └── Apache Flink ──► ClickHouse / Prometheus ──► Grafana Dashboards
   │             • P50 / P95 / P99 TTFT & ITL
   │             • Cost per tenant / model
   │             • Error rates & token counts
   │
   └──► [Route B: Tiered Evaluation Sampling Path]
           ├── 100% Deterministic Scanners (Toxicity regex, PII regex, format checks)
           ├── 1% Stratified Sampling for LLM-as-a-Judge (1M evals/day)
           └── 100% of User Thumbs-Down / Reported Failures
```

### 3. The Continuous Evaluation Worker Pool (LLM-as-a-Judge)

```text
Sampled Traces (1M requests/day)
   │
   ▼
Evaluator Queue (SQS / Celery)
   │
   ▼
Worker Fleet (Powered by lightweight quantized 8B Judge Models)
   │
   ├── Metric 1: Faithfulness (Does context entail answer?)
   ├── Metric 2: Context Relevance (Is retrieved text relevant?)
   ├── Metric 3: Answer Semantic Drift against Golden Set
   │
   ▼
Alerting Engine (PagerDuty / Slack)
   └── Triggers alert if:
         • Faithfulness drops below 0.95 in a 1-hour window
         • Jailbreak bypass classifier fires
         • Cost per department exceeds daily allocation
```

### 4. Enterprise FinOps & Cost Attribution
- Every request carries an `X-Tenant-ID`, `X-Cost-Center`, and `X-Application-ID` header.
- ClickHouse aggregates token usage using exact model rate cards (e.g., Llama-3-70B = \$0.0007/1K tokens, GPT-4o = \$0.005/1K tokens).
- Teams receive real-time cost transparency dashboards and automated hard budget cutoffs.

---

### 💡 Summary Table: Quick Whiteboard Comparison

| System | Primary Bottleneck | Golden Metric | Core Architectural Weapon |
| :--- | :--- | :--- | :--- |
| **1. Code Copilot** | Latency & Context Relevance | P95 Latency < 80ms | FIM Tokens + AST Slicing + Speculative 1B Drafter |
| **2. Real-Time Voice** | Turn-taking Latency | Glass-to-Glass < 300ms | WebRTC + Streaming VAD + Token-chunked TTS |
| **3. Enterprise RAG** | Access Control & Recall | Zero Unauthorized Access | Pre-filtered HNSW ACL Bitmasks + Redis Permission Cache |
| **4. GenAI Telemetry** | High Ingestion Volume & Judge Cost | 4,000 RPS at Low Cost | Kafka + Flink + Stratified 1% Sampling to ClickHouse |
