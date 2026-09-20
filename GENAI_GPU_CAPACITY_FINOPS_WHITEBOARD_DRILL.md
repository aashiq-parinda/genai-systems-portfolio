# ⚡ The MAANG GenAI GPU Capacity & FinOps Whiteboard Drill
## 🧮 Mental Arithmetic, VRAM Sizing Formulas & TCO Breakeven Proofs for Infrastructure Interviews

> 🎯 **Target Stage**: GenAI Systems Design, Infrastructure Sizing & Back-of-the-Envelope Whiteboard Arithmetic  
> 🏷️ **Target Roles**: GenAI Infrastructure / MLOps Engineer | Staff AI Systems Architect | Applied AI Scientist  
> 🏢 **Target Companies**: Meta, Google (GCP), Amazon (AWS Bedrock), Apple, Netflix, OpenAI, Nebius, RunPod  
> 📌 **Companion Guides**: [🏛️ System Design Whiteboard Playbook](GENAI_SYSTEM_DESIGN_WHITEBOARD_PLAYBOOK.md) | [🎙️ Verbal Tech Guide](GENAI_VERBAL_TECHNICAL_INTERVIEW_GUIDE.md) | [📐 GPT Math Formulas](GPT_MATHEMATICAL_FORMULAS_EXPLAINED.md)

---

## ⚡ The Whiteboard Math Rule: Zero Calculators Allowed

> [!IMPORTANT]
> In MAANG infrastructure rounds, interviewers do not expect 8 decimal places. They evaluate your ability to do **rapid, orders-of-magnitude dimensional analysis** on a whiteboard.
> 
> When an interviewer asks: *"How many H100s do we need for 1,000 concurrent users on a 70B model?"*, candidates who freeze or guess fail. Candidates who write down the **3 Golden Laws of LLM Memory** and execute mental arithmetic in 90 seconds get instant L6/Staff offers.

---

## 📌 Table of Contents

1. [📐 The 3 Golden Formulas of LLM Memory Sizing](#1️⃣-the-3-golden-formulas-of-llm-memory-sizing)
2. [🧮 Whiteboard Drill 1: Sizing an H100 Cluster for Llama-3-70B](#2️⃣-whiteboard-drill-1-sizing-an-h100-cluster-for-llama-3-70b)
3. [⚡ Whiteboard Drill 2: Calculating Theoretical Max Tokens/Sec (The Roofline Model)](#3️⃣-whiteboard-drill-2-calculating-theoretical-max-tokenssec)
4. [💰 Whiteboard Drill 3: Cloud API vs. Self-Hosted H100 TCO Breakeven Analysis](#4️⃣-whiteboard-drill-3-cloud-api-vs-self-hosted-h100-tco-breakeven-analysis)
5. [📊 Mental Math Conversion Cheat Sheet](#5️⃣-mental-math-conversion-cheat-sheet)

---

## 1️⃣ The 3 Golden Formulas of LLM Memory Sizing

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│               TOTAL VRAM REQUIRED AT INFERENCE TIME                         │
│                                                                             │
│ VRAM_Total = VRAM_Weights + VRAM_KV_Cache + VRAM_Activations + CUDA_Context │
│                                                                             │
│  [ Weights ]         [ KV Cache ]           [ Temp Activations & CUDA ]     │
│   P × Bytes           2 × L × H_kv × D × B        ~15-20% Safety Buffer     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Formula 1: Model Weight VRAM
$$M_{\text{weights}} = \text{Parameters } (P) \times \text{Bytes per Parameter}$$

| Precision | Bits per Param | Bytes per Param | Example: 7B Model | Example: 70B Model |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Full)** | 32 bits | 4.0 Bytes | 28 GB | 280 GB |
| **FP16 / BF16 (Standard)** | 16 bits | 2.0 Bytes | 14 GB | 140 GB |
| **FP8 / INT8 (Serving)** | 8 bits | 1.0 Byte | 7 GB | 70 GB |
| **INT4 / NF4 (Quantized)** | 4 bits | 0.5 Bytes | 3.5 GB | 35 GB |

---

### Formula 2: The KV Cache Memory Formula (Per Token)
For each token in the context window across a batch of requests, we store Key and Value vectors:
$$M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads(kv)}} \times d_{\text{head}} \times \text{Bytes} \times S \times B$$

- $2$: For Keys and Values.
- $n_{\text{layers}}$: Number of Transformer layers.
- $n_{\text{heads(kv)}}$: Number of Key/Value attention heads (Equal to Query heads in MHA; much smaller in GQA).
- $d_{\text{head}}$: Head dimension (typically 128).
- $\text{Bytes}$: Usually 2 bytes (FP16/BF16) or 1 byte (FP8 KV Cache).
- $S$: Context length (prompt + generated tokens).
- $B$: Batch size (concurrent active sequences).

> [!TIP]
> **Staff-Level Shortcut Rule of Thumb**:
> - For **Llama-3-8B** (GQA: 8 KV heads, 32 layers, $d=128$, FP16): **~0.13 MB per token** $\approx 1.3 \text{ GB}$ per 10,000 tokens.
> - For **Llama-3-70B** (GQA: 8 KV heads, 80 layers, $d=128$, FP16): **~0.33 MB per token** $\approx 3.3 \text{ GB}$ per 10,000 tokens.

---

## 2️⃣ Whiteboard Drill 1: Sizing an H100 Cluster for Llama-3-70B

### ❓ The Interviewer Asks:
> *"We want to serve Llama-3-70B (FP16) on NVIDIA H100 GPUs (80GB SXM5). We expect a steady concurrency of $B = 256$ active users with an average context length of $S = 4{,}096$ tokens. How many H100 GPUs do we need at minimum?"*

---

### 🗣️ Step-by-Step Whiteboard Mental Math:

#### Step 1: Model Weight Memory
$$M_{\text{weights}} = 70 \times 10^9 \text{ params} \times 2 \text{ Bytes (FP16)} = 140 \text{ GB}$$

#### Step 2: KV Cache Memory for 256 Concurrent Streams at 4K Context
- Total tokens in flight = $256 \text{ sequences} \times 4{,}096 \text{ tokens} \approx 1{,}048{,}576 \text{ tokens} \approx 1 \times 10^6 \text{ tokens}$.
- Llama-3-70B Architecture specs: 80 layers, 8 KV heads, head dim 128, FP16 (2 bytes):
  $$\text{KV bytes per token} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 0.328 \text{ MB}$$
- Total KV Cache Memory:
  $$M_{\text{KV}} = 1{,}048{,}576 \text{ tokens} \times 0.328 \text{ MB} \approx 344 \text{ GB}$$

#### Step 3: Total VRAM & Safety Buffer
- Raw Total = $140 \text{ GB (Weights)} + 344 \text{ GB (KV Cache)} = 484 \text{ GB}$.
- Add 20% buffer for CUDA runtime, temporary activation tensors, and memory fragmentation:
  $$M_{\text{total}} = 484 \times 1.20 \approx 580 \text{ GB}$$

#### Step 4: Number of H100 GPUs (80GB each)
$$\text{Num GPUs} = \frac{580 \text{ GB}}{80 \text{ GB / GPU}} = 7.25 \text{ GPUs}$$

#### Step 5: Distributed Topology & Hardware Constraint:
> "Because GPU cluster topologies require powers of 2 for Tensor Parallelism (TP) over NVLink (TP=1, 2, 4, 8), **we need exactly 1 node of 8x H100 GPUs (640 GB total VRAM)**.
> 
> Running **Tensor Parallelism TP=8** splits the 140 GB model across the 8 GPUs (~17.5 GB weights per GPU), leaving ~62.5 GB of free VRAM on each GPU exclusively for KV Cache, which comfortably handles our 256 concurrent streams with room to spare!"

---

## 3️⃣ Whiteboard Drill 2: Calculating Theoretical Max Tokens/Sec

### ❓ The Interviewer Asks:
> *"What is the maximum theoretical generation speed (tokens/sec) for a single user querying an unquantized 70B model on an H100 SXM5 GPU? Why can't it run at 1,000 tokens/sec?"*

---

### 🗣️ Step-by-Step Whiteboard Mental Math:

1. **Identify the Bottleneck**:
   - Generating 1 token during the **Decode phase** is memory-bandwidth bound.
   - To emit a single token, the GPU must stream the entire weight matrix from High-Bandwidth Memory (HBM) into SRAM.
2. **Hardware Specifications**:
   - An NVIDIA H100 SXM5 has an HBM3 memory bandwidth of **$3.35 \text{ TB/s}$** ($3{,}350 \text{ GB/s}$).
3. **Weight Size**:
   - A 70B parameter model in FP16 weighs **$140 \text{ GB}$**.
4. **Theoretical Single-User Ceiling**:
   $$\text{Max Tokens / Sec} = \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Weight Size (GB)}} = \frac{3{,}350 \text{ GB/s}}{140 \text{ GB}} \approx 23.9 \text{ tokens / second}$$

> **Key Interview Takeaway**:
> "Even on the world's most powerful H100 GPU, a single user running pure FP16 Llama-3-70B on 1 GPU can never exceed ~24 tokens/sec without techniques like:
> 1. **Tensor Parallelism (TP=4 or TP=8)**: Aggregating memory bandwidth across 8 GPUs ($8 \times 3.35 \text{ TB/s} = 26.8 \text{ TB/s} \to 190 \text{ tokens/sec}$).
> 2. **Quantization (INT4 / FP8)**: Halving or quartering model weight transfer size.
> 3. **Speculative Decoding**: Emitting multiple verified tokens per memory-read pass."

---

## 4️⃣ Whiteboard Drill 3: Cloud API vs. Self-Hosted H100 TCO Breakeven Analysis

### ❓ The Interviewer Asks:
> *"Our company serves 50 million tokens per day (70% input, 30% output). Should we use OpenAI GPT-4o via API or rent an 8x H100 server on AWS/Nebius to host Llama-3-70B? What is our monthly FinOps breakeven point?"*

---

### 🗣️ Step-by-Step Whiteboard Mental Math:

#### Option A: Cloud API Cost (OpenAI GPT-4o)
- **Daily Volume**: 50M tokens.
  - Input tokens (70%): $35\text{M tokens / day} \times \$2.50 / \text{M} = \$87.50 / \text{day}$.
  - Output tokens (30%): $15\text{M tokens / day} \times \$10.00 / \text{M} = \$150.00 / \text{day}$.
- **Total API Daily Cost** = $\$87.50 + \$150.00 = \$237.50 / \text{day}$.
- **Monthly API Cost (30 days)**:
  $$\text{Cost}_{\text{API}} = \$237.50 \times 30 \approx \$7{,}125 / \text{month}$$

---

#### Option B: Self-Hosted Cloud Infrastructure (8x H100 SXM5 Node)
- **Server Rental Rate**: ~$2.50 to $3.00 per H100 GPU hour.
  - 8 GPUs $\times \$2.75 / \text{GPU hr} = \$22.00 / \text{server hour}$.
- **Monthly Server Cost (24/7 reservation)**:
  $$\text{Server Cost} = \$22.00 \times 24 \text{ hrs} \times 30 \text{ days} = \$15{,}840 / \text{month}$$
- **Add MLOps & Networking Overhead (+20%)**:
  $$\text{Cost}_{\text{Infra}} \approx \$15{,}840 \times 1.20 \approx \$19{,}000 / \text{month}$$

---

#### ⚖️ The Breakeven Verdict:
$$\frac{\text{Monthly Infra Cost (\$19,000)}}{\text{Daily API Cost (\$237.50 / day)}} \approx 80 \text{ days}$$
$$\text{Breakeven Daily Token Volume} = 50\text{M tokens} \times \left(\frac{\$19{,}000}{\$7{,}125}\right) \approx \mathbf{133\text{ Million Tokens / Day}}$$

> **What to Say to the Interviewer**:
> "At 50M tokens/day, **calling the API is more than 2.6x cheaper** (\$7.1K vs \$19K) and carries zero infrastructure maintenance overhead. 
> 
> **We should only self-host 8x H100s if**:
> 1. Our daily volume crosses **133 Million tokens/day**, where self-hosting becomes cheaper per token.
> 2. Strict data privacy or compliance (HIPAA / air-gapped banking) forbids third-party API transmission.
> 3. We require custom LoRA adapter multiplexing or deterministic sub-50ms latency guarantees."

---

## 5️⃣ Mental Math Conversion Cheat Sheet

Memorize these constants for lightning whiteboard delivery:

| Metric | Constant | Mental Shortcut |
| :--- | :--- | :--- |
| **H100 SXM5 VRAM** | 80 GB HBM3 | Round to 80 GB |
| **H100 Memory Bandwidth** | 3.35 TB/s | Multiply by 3.3 TB/s |
| **A100 SXM4 VRAM** | 80 GB HBM2e | 2.0 TB/s Bandwidth |
| **L70B Weight (FP16)** | 140 GB | $70 \times 2$ |
| **L70B Weight (FP8)** | 70 GB | $70 \times 1$ |
| **L70B Weight (INT4)** | 35 GB | $70 \times 0.5$ |
| **KV Cache per 1M Tokens (L70B GQA)** | ~330 GB | 1 token $\approx 0.33$ KB |
| **Hours in a Month** | 730 hours | Round to $24 \times 30 = 720$ |
| **1 Token in English** | ~0.75 words | 4 characters |
