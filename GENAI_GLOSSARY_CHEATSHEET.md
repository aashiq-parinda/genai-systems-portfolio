# 📚 The Ultimate Generative AI & LLM Engineering Glossary Cheat Sheet

> 🎯 **Target Roles**: GenAI Application Engineer | Applied AI Scientist | GenAI Infrastructure & MLOps Engineer | Lead / Principal AI Architect  
> 🏢 **Target Companies**: Meta | Google (DeepMind/GCP) | Amazon (AWS/Bedrock) | Apple | Netflix | OpenAI  
> 📌 **Companion Repository**: [Cracking The GenAI Portfolio](README.md) | [MAANG GenAI Interview Guide](MAANG_GENAI_INTERVIEW_GUIDE.md)  

---

## ⚡ 10-Second Executive Summary

This glossary is engineered specifically for **MAANG GenAI technical interviews** — spanning from **Level 0 Basics & Fundamentals** to **Level 3 Distributed Infrastructure**. 

Each term is structured into 3 high-impact parts:
1. ⚡ **1-Sentence Definition** (for instant recall & basic understanding).
2. 🧠 **MAANG Systems Mechanics** (the exact technical detail interviewers look for).
3. 💡 **Interview Gotcha & Trade-Off** (to prove senior engineering depth).

---

## 📌 Table of Contents
- [🌱 0. Core AI & Machine Learning Fundamentals (Level 0 Basics)](#-0-core-ai--machine-learning-fundamentals-level-0-basics)
- [🧠 1. LLM Architecture & Attention Mechanisms](#-1-llm-architecture--attention-mechanisms)
- [⚡ 2. Serving Infrastructure & Latency Metrics](#-2-serving-infrastructure--latency-metrics)
- [🔬 3. Fine-Tuning, Alignment & Post-Training](#-3-fine-tuning-alignment--post-training)
- [🧮 4. Quantization & Model Compression](#-4-quantization--model-compression)
- [🔍 5. RAG, Vector Search & Knowledge Grounding](#-5-rag-vector-search--knowledge-grounding)
- [🤖 6. Multi-Agent Systems & Tool Calling](#-6-multi-agent-systems--tool-calling)
- [🌐 7. Distributed Training & Parallelism](#-7-distributed-training--parallelism)

---

## 🌱 0. Core AI & Machine Learning Fundamentals (Level 0 Basics)

---

### 1. Tokenization (BPE / SentencePiece / WordPiece)
* ⚡ **1-Sentence Definition**: The process of breaking raw input text strings into discrete numerical chunks (tokens) that a neural network can process.
* 🧠 **MAANG Systems Mechanics**: Modern LLMs use **Byte-Pair Encoding (BPE)** or **SentencePiece**. BPE starts with individual bytes/characters and iteratively merges the most frequently occurring adjacent pair of tokens into a single sub-word token until vocabulary size $V$ is reached (e.g., Llama-3 vocabulary size $V = 128,256$).
* 💡 **Interview Gotcha**: Tokenization causes strange LLM edge cases (e.g., struggling with letter counting in "strawberry" or trailing whitespace syntax errors) because words are split into sub-word tokens rather than individual letters or whole words.

---

### 2. Word Embedding Vector & Vector Space
* ⚡ **1-Sentence Definition**: A high-dimensional numerical representation of a token or text chunk in continuous vector space where semantically similar items sit close together.
* 🧠 **MAANG Systems Mechanics**: Maps discrete token index $t_i \in \{1, \dots, V\}$ to a continuous vector $v \in \mathbb{R}^d$ via an embedding lookup matrix $E \in \mathbb{R}^{V \times d}$ (where $d$ is model dimension, e.g., $d = 4096$).
* 💡 **Interview Gotcha**: Token embeddings capture static semantic relationships, but require Transformer **Attention layers** to dynamically contextualize words based on surrounding context.

---

### 3. Logits & Softmax Function
* ⚡ **1-Sentence Definition**: Logits are the unnormalized raw output scores produced by the model's final linear layer, which Softmax converts into a normalized probability distribution.
* 🧠 **MAANG Systems Mechanics**: The Softmax function converts logit vector $z_i$ into probabilities $P(y_i)$:

$$P(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $T$ is the **Sampling Temperature**.
* 💡 **Interview Gotcha**: Setting $T \to 0$ approaches greedy deterministic decoding ($\arg\max$), whereas higher $T > 1.0$ flattens the probability distribution, encouraging creative/diverse outputs at the risk of hallucinations.

---

### 4. Loss Function & Cross-Entropy Loss
* ⚡ **1-Sentence Definition**: A mathematical function measuring the error difference between the model's predicted probability distribution and the ground-truth target tokens.
* 🧠 **MAANG Systems Mechanics**: Language models are trained using **Negative Log-Likelihood (Cross-Entropy Loss)** over token sequences:

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{S} \log P(y_i^* \mid y_{<i})$$

* 💡 **Interview Gotcha**: Loss measures training progress, but lower loss does not strictly guarantee zero hallucinations or better human alignment (hence the need for post-training RLHF/DPO!).

---

### 5. Gradient Descent & AdamW Optimizer
* ⚡ **1-Sentence Definition**: Gradient Descent updates model parameters in the direction that minimizes loss, while AdamW enhances SGD with adaptive learning rates and decoupled weight decay.
* 🧠 **MAANG Systems Mechanics**: AdamW maintains two tracking states per parameter: the 1st moment (mean of gradients $m_t$) and 2nd moment (uncentered variance of gradients $v_t$). This requires **8 bytes of FP32 optimizer memory per parameter**!
* 💡 **Interview Gotcha**: Decoupling weight decay ($w_{t+1} = w_t - \gamma \lambda w_t$) in AdamW prevents weight magnitudes from exploding during large-scale pre-training.

---

### 6. Bias-Variance Tradeoff (Overfitting vs. Underfitting)
* ⚡ **1-Sentence Definition**: Bias is error from over-simplified assumptions (underfitting), while Variance is error from sensitivity to small fluctuations in training data (overfitting).
* 🧠 **MAANG Systems Mechanics**: Overfitting in LLMs occurs when memorization dominates generalization. Mitigated via weight decay, dropout, dataset deduplication (MinHash LSH), and early stopping.
* 💡 **Interview Gotcha**: In LLM fine-tuning, training beyond 3-5 epochs on small datasets causes memorization of exact training phrases, destroying out-of-distribution reasoning.

---

### 7. Cosine Similarity vs. Euclidean Distance vs. Dot Product
* ⚡ **1-Sentence Definition**: Mathematical metrics used in Vector Databases to measure similarity between query embeddings and document embeddings.
* 🧠 **MAANG Systems Mechanics**:
  * **Cosine Similarity**: Measures the cosine of the angle between two vectors:

$$\text{Cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

  * **Dot Product**: Equivalent to Cosine Similarity when vectors are L2-normalized ($\|A\| = \|B\| = 1.0$), but **10x faster to compute** on GPUs!
* 💡 **Interview Gotcha**: Always normalize embeddings prior to indexing in Qdrant/Milvus to convert expensive Cosine Similarity lookups into ultra-fast GPU matrix Dot Product multiplications!

---

### 8. Auto-Regressive Models vs. Encoder-Only vs. Encoder-Decoder
* ⚡ **1-Sentence Definition**: The three primary Transformer architectural paradigms tailored for generation, classification, or translation.
* 🧠 **MAANG Systems Mechanics**:
  * **Decoder-Only (Auto-Regressive, e.g., Llama-3, GPT-4o)**: Uses causal masking so token $i$ can only attend to past tokens $1 \dots i-1$. Optimized for text generation.
  * **Encoder-Only (e.g., BERT, DeBERTa-v3)**: Uses bi-directional attention so tokens attend to all sequence tokens. Optimized for classification and embeddings.
  * **Encoder-Decoder (e.g., T5, Whisper)**: Bi-directional encoder coupled with causal decoder. Optimized for translation and summarization.
* 💡 **Interview Gotcha**: Modern frontier LLMs have unified around **Decoder-Only** architecture because causal pre-training scales most predictably with compute and data scaling laws.

---

### 9. Prompt Engineering (Zero-Shot, Few-Shot, Chain-of-Thought)
* ⚡ **1-Sentence Definition**: In-context steering techniques that guide LLM responses without updating model weights.
* 🧠 **MAANG Systems Mechanics**:
  * **Zero-Shot**: Direct prompt without examples.
  * **Few-Shot**: Including $k$ exemplar input-output pairs in the context window.
  * **Chain-of-Thought (CoT)**: Prompting the model to generate intermediate step-by-step reasoning tokens before providing a final answer ("Think step by step").
* 💡 **Interview Gotcha**: CoT works because generating intermediate reasoning tokens expands the model's compute budget per query (each token forward pass executes $2P$ FLOPs of computation!).

---

### 10. Hallucination & Factual Grounding
* ⚡ **1-Sentence Definition**: A hallucination is when an LLM generates plausible-sounding but factually incorrect or ungrounded assertions.
* 🧠 **MAANG Systems Mechanics**: Hallucinations stem from the next-token prediction objective: the model optimizes for statistical token likelihood, not factual verification. Mitigated via Retrieval-Augmented Generation (RAG) and NLI entailment guardrails.
* 💡 **Interview Gotcha**: Hallucination cannot be 100% eliminated in unconstrained auto-regressive generation, but can be bounded to <1% using strict context-grounded RAG pipelines.

---

## 🧠 1. LLM Architecture & Attention Mechanisms

---

### 11. Multi-Head Attention (MHA)
* ⚡ **1-Sentence Definition**: The core transformer mechanism that projects query, key, and value vectors into $H$ independent sub-spaces to attend to different sequence tokens simultaneously.
* 🧠 **MAANG Systems Mechanics**: Maintains $H$ distinct Key ($K$) and Value ($V$) heads per layer. Every query head has its own corresponding $K$ and $V$ head ($1:1$ ratio), requiring $2 \times L \times H \times D \times S \times B \times b$ bytes of GPU memory for KV caching.
* 💡 **Interview Gotcha**: MHA causes severe memory bandwidth bottlenecks during inference decode phase because loading large $KV$ matrices into VRAM per token exceeds GPU memory bandwidth capacity.

---

### 12. Grouped-Query Attention (GQA)
* ⚡ **1-Sentence Definition**: An attention variant that groups multiple query heads to share a single Key-Value head, drastically reducing KV cache size.
* 🧠 **MAANG Systems Mechanics**: In Llama-3-70B, $H_{\text{query}} = 64$ while $H_{\text{KV}} = 8$ (an 8:1 grouping ratio). This slashes KV cache memory footprint by **8x** compared to standard MHA.
* 💡 **Interview Gotcha**: GQA achieves near-identical quality to MHA while allowing up to **8x larger batch sizes** or context lengths during serving.

---

### 13. Rotary Position Embeddings (RoPE)
* ⚡ **1-Sentence Definition**: A positional encoding technique that encodes absolute position with a rotation matrix and naturally incorporates relative position in attention inner products.
* 🧠 **MAANG Systems Mechanics**: Applies a 2D rotation matrix to pairs of key/query vector coordinates:

$$\mathbf{R}_{\Theta, m}^d \mathbf{x}_m$$

This allows attention scores between token $m$ and token $n$ to depend strictly on their relative distance $(m - n)$.
* 💡 **Interview Gotcha**: Enables context length extrapolation via RoPE scaling techniques (e.g., YaRN, NTK-aware scaling) without retraining base model attention layers from scratch.

---

### 14. FlashAttention (v1 / v2 / v3)
* ⚡ **1-Sentence Definition**: An IO-aware exact attention algorithm that speeds up attention computation and reduces memory usage from $O(N^2)$ to $O(N)$ by tiling matrix blocks.
* 🧠 **MAANG Systems Mechanics**: Avoids materializing the massive $N \times N$ attention weight matrix in slow GPU High Bandwidth Memory (HBM). Instead, it fuses softmax and matrix multiplication into fast GPU SRAM (L1 cache) using online softmax rescaling.
* 💡 **Interview Gotcha**: FlashAttention-3 leverages NVIDIA H100 Hopper Tensor Core asynchronous warp-group instructions to overlap GEMM operations with memory loading, achieving >75% of peak TFLOPs.

---

## ⚡ 2. Serving Infrastructure & Latency Metrics

---

### 15. Time-To-First-Token (TTFT)
* ⚡ **1-Sentence Definition**: The total latency elapsed from sending an API prompt to receiving the very first output token stream chunk.
* 🧠 **MAANG Systems Mechanics**: TTFT is dominated by the **Prefill Phase** of inference, which is compute-bound ($2 \times P \times S_{\text{prompt}}$ FLOPs executed concurrently across prompt tokens).
* 💡 **Interview Gotcha**: Optimize TTFT by using semantic caching, speculative prefill, or prompt lookup decoding.

---

### 16. Inter-Token Latency (ITL)
* ⚡ **1-Sentence Definition**: The time delay between generating consecutive output tokens during streaming.
* 🧠 **MAANG Systems Mechanics**: ITL is dominated by the **Decode Phase** of inference, which is memory-bandwidth bound (reading all model parameters from GPU HBM to Tensor Cores per single generated token).
* 💡 **Interview Gotcha**: P99 ITL spikes are usually caused by KV cache memory allocation pauses or continuous batching queue preemptions.

---

### 17. PagedAttention
* ⚡ **1-Sentence Definition**: An attention algorithm (pioneered by vLLM) that manages KV cache memory using virtual memory paging concepts from Operating Systems.
* 🧠 **MAANG Systems Mechanics**: Allocates KV cache in fixed-size physical block pages (e.g., 16 tokens/page). Maintains a dynamic block mapping table to eliminate contiguous VRAM allocation constraints.
* 💡 **Interview Gotcha**: Slashes GPU memory waste from fragmented contiguous allocations from **>60% down to under 4%**, quadrupling batch throughput.

---

### 18. Continuous Batching (Iteration-Level Scheduling)
* ⚡ **1-Sentence Definition**: An inference iteration scheduling mechanism that dynamically injects new queries into an active batch as soon as old queries emit `<EOS>`.
* 🧠 **MAANG Systems Mechanics**: Instead of waiting for an entire static batch to finish decoding all sequences, continuous batching reschedules active GPU memory at every token iteration step.
* 💡 **Interview Gotcha**: Completely eliminates GPU compute idle time caused by sequence length variance in static batching.

---

### 19. Speculative Decoding
* ⚡ **1-Sentence Definition**: A latency optimization technique where a fast, small Draft Model rapidly proposes candidate tokens, which a large Target Model verifies in parallel.
* 🧠 **MAANG Systems Mechanics**: Executes a single parallel forward pass over $K$ draft tokens on the large target model. Uses a modified rejection sampling algorithm ($P_{\text{accept}} = \min(1, p(x)/q(x))$) to guarantee identical output probability distributions.
* 💡 **Interview Gotcha**: Delivers **2x to 4x latency reductions** without degrading model quality by a single bit!

---

## 🔬 3. Fine-Tuning, Alignment & Post-Training

---

### 20. Low-Rank Adaptation (LoRA)
* ⚡ **1-Sentence Definition**: A parameter-efficient fine-tuning (PEFT) method that freezes base model weights $W_0$ and injects trainable rank-decomposition matrices $B \cdot A$.
* 🧠 **MAANG Systems Mechanics**: Replaces full parameter update $\Delta W$ with two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ where rank $r \ll \min(d, k)$:

$$h = W_0 x + \frac{\alpha}{r} (B A) x$$

* 💡 **Interview Gotcha**: Keeps base model frozen, allowing 100+ tenant-specific adapters to be swapped dynamically in VRAM at zero base-model loading cost.

---

### 21. QLoRA (Quantized LoRA)
* ⚡ **1-Sentence Definition**: An extension of LoRA that quantizes the frozen base model to 4-bit NormalFloat4 (NF4) while fine-tuning 16-bit LoRA adapter parameters.
* 🧠 **MAANG Systems Mechanics**: Introduces NormalFloat4 (information-theoretically optimal quantile distribution), Double Quantization (saving 0.37 bits/param), and Paged Optimizers (using CUDA unified memory to page VRAM spikes to CPU RAM).
* 💡 **Interview Gotcha**: Reduces fine-tuning VRAM footprint of a 70B model from **1,120 GB down to 41.4 GB**, enabling 70B fine-tuning on a single A100 80GB GPU!

---

### 22. Direct Preference Optimization (DPO)
* ⚡ **1-Sentence Definition**: An alignment algorithm that directly optimizes policy weights using human preference pairs without training an explicit Reward Model or using PPO.
* 🧠 **MAANG Systems Mechanics**: Derives a closed-form implicit reward function from policy probabilities:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

* 💡 **Interview Gotcha**: Eliminates the training instability, actor-critic complexity, and high memory overhead of online PPO rollouts.

---

### 23. Group Relative Policy Optimization (GRPO)
* ⚡ **1-Sentence Definition**: A reinforcement learning algorithm (used in DeepSeek-R1) that calculates relative reward advantage across a group of sampled outputs instead of training a Critic model.
* 🧠 **MAANG Systems Mechanics**: For each prompt $q$, samples $G$ outputs $\{o_1, \dots, o_G\}$ and computes normalized advantage:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

* 💡 **Interview Gotcha**: Slashing the requirement for a separate Value (Critic) model saves ~50% GPU VRAM during RL training at scale.

---

## 🧮 4. Quantization & Model Compression

---

### 24. Activation-Aware Weight Quantization (AWQ)
* ⚡ **1-Sentence Definition**: An INT4 post-training quantization method that protects the top 1% salient weight channels based on activation magnitude.
* 🧠 **MAANG Systems Mechanics**: Multiplies salient weight channels by a scaling factor $s \ge 1$ while inversely scaling activation inputs $X$ by $s^{-1}$, keeping Tensor Core INT4 matrix multiplication hardware-uniform.
* 💡 **Interview Gotcha**: Outperforms GPTQ on low-bit quantization accuracy because it observes input activations rather than relying purely on weight matrices.

---

### 25. FP8 Precision (E4M3 vs. E5M2)
* ⚡ **1-Sentence Definition**: 8-bit floating point formats supported natively on modern NVIDIA Hopper (H100) Tensor Cores.
* 🧠 **MAANG Systems Mechanics**:
  * **E4M3** (1 sign bit, 4 exponent bits, 3 mantissa bits): Higher precision, used for forward pass weight and activation GEMMs.
  * **E5M2** (1 sign bit, 5 exponent bits, 2 mantissa bits): Higher dynamic range, used for backward pass gradient computation.
* 💡 **Interview Gotcha**: Delivers **2x GEMM throughput** over FP16 on H100 GPUs with virtually zero loss in perplexity.

---

## 🔍 5. RAG, Vector Search & Knowledge Grounding

---

### 26. Reciprocal Rank Fusion (RRF)
* ⚡ **1-Sentence Definition**: An algorithm that combines search results from multiple search retrieval systems (e.g., BM25 keyword + Dense Vector search) without score normalization.
* 🧠 **MAANG Systems Mechanics**: Computes unified document rank score using position reciprocal formula:

$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

where $k = 60$.
* 💡 **Interview Gotcha**: RRF eliminates the need to calibrate arbitrary distance metrics (e.g., BM25 un-bounded scores vs Cosine similarity $[-1, 1]$).

---

### 27. HNSW (Hierarchical Navigable Small World)
* ⚡ **1-Sentence Definition**: A multi-layer graph-based approximate nearest neighbor (ANN) vector index that provides ultra-fast logarithmic search time complexity.
* 🧠 **MAANG Systems Mechanics**: Organizes vector embeddings into hierarchical skip-list graph layers. Top layers contain long-distance highway links, while bottom layers contain dense local neighbor links.
* 💡 **Interview Gotcha**: Extremely fast search latency ($<5\text{ms}$ over 10M vectors), but consumes significant RAM memory during index construction.

---

## 🤖 6. Multi-Agent Systems & Tool Calling

---

### 28. ReAct Pattern (Reason + Act)
* ⚡ **1-Sentence Definition**: An agent prompting framework that interleaves reasoning trace generation ("Thought") with domain action execution ("Act") and environment observation ("Obs").
* 🧠 **MAANG Systems Mechanics**: Formats agent state history into structured loops:

```text
Thought ──► Action (Tool Call) ──► Observation (Result) ──► Final Answer
```

* 💡 **Interview Gotcha**: Susceptible to infinite looping; must be guarded by deterministic state hash history tracking (SHA256 of state diffs).

---

## 🌐 7. Distributed Training & Parallelism

---

### 29. 3D Parallelism (TP + PP + DP)
* ⚡ **1-Sentence Definition**: The combination of Tensor Parallelism, Pipeline Parallelism, and Data Parallelism to scale massive models across thousands of GPUs.
* 🧠 **MAANG Systems Mechanics**:
  * **Tensor Parallelism (TP)**: Splits intra-layer weight matrices across GPUs within an 8-GPU NVLink node.
  * **Pipeline Parallelism (PP)**: Splits model layers sequentially across nodes over InfiniBand network switches.
  * **Data Parallelism (DP/ZeRO-3)**: Replicates micro-batches and partitions optimizer state across data parallel ranks.
* 💡 **Interview Gotcha**: Always keep TP bounded to single NVLink nodes (TP=8) because TP requires ultra-high all-reduce interconnect bandwidth ($900\text{ GB/s}$).

---
*This cheat sheet is part of the [Cracking The GenAI Portfolio](README.md) open-source repository.*
