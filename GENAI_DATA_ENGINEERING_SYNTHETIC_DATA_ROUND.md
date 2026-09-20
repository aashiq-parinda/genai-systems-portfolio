# 🧬 The MAANG GenAI Data Engineering & Synthetic Data Interview Guide
## 🔬 Pre-Training Curation, MinHash LSH Deduplication, Synthetic Evol-Instruct & Benchmark Decontamination

> 🎯 **Target Stage**: Data Curation Round, Applied Scientist Post-Training Interview, GenAI Data Platform System Design  
> 🏷️ **Target Roles**: Applied AI Scientist | GenAI Data Engineer | Post-Training / Alignment Engineer | AI Research Engineer  
> 🏢 **Target Companies**: Meta (Llama Data Team), OpenAI, Anthropic, Databricks (MosaicML), Google DeepMind, Mistral  
> 📌 **Companion Guides**: [🏛️ System Design Whiteboard](GENAI_SYSTEM_DESIGN_WHITEBOARD_PLAYBOOK.md) | [⚡ GPU Capacity Drill](GENAI_GPU_CAPACITY_FINOPS_WHITEBOARD_DRILL.md) | [🎙️ Verbal Tech Guide](GENAI_VERBAL_TECHNICAL_INTERVIEW_GUIDE.md)

---

## ⚡ The Modern Reality: "Architecture Is a Commodity; Data Is the Moat"

> [!IMPORTANT]
> Model architectures across top labs have largely converged: **Transformer Decoder + RoPE + RMSNorm + SwiGLU + GQA**. 
> 
> What separates a mediocre model from state-of-the-art models like **Llama-3, Claude 3.5 Sonnet, and DeepSeek-V3** is **Data Engineering**: trillion-token deduplication, perplexity filtering, multi-mutation synthetic instruction synthesis, and rigorous decontamination.
> 
> If you walk into a MAANG research or post-training interview and cannot explain **MinHash LSH, Evol-Instruct, Rejection Sampling, and Process Reward Models (PRM)**, you will not pass.

---

## 📌 Table of Contents

1. [🏭 The End-to-End Trillion-Token Pre-Training Data Pipeline](#1️⃣-the-end-to-end-trillion-token-pre-training-data-pipeline)
2. [🔍 Fuzzy Deduplication: MinHash & Locality-Sensitive Hashing (LSH) Math](#2️⃣-fuzzy-deduplication-minhash--locality-sensitive-hashing-lsh-math)
3. [🧪 Synthetic Data Generation: Evol-Instruct, Self-Instruct & UltraFeedback](#3️⃣-synthetic-data-generation-evol-instruct-self-instruct--ultrafeedback)
4. [🎯 Post-Training Rejection Sampling & Process Reward Models (PRMs)](#4️⃣-post-training-rejection-sampling--process-reward-models-prms)
5. [🛡️ Benchmark Decontamination & Test-Set Leakage Prevention](#5️⃣-benchmark-decontamination--test-set-leakage-prevention)
6. [🎙️ Top 3 Data Engineering Interview Scenarios & Model Answers](#6️⃣-top-3-data-engineering-interview-scenarios--model-answers)

---

## 1️⃣ The End-to-End Trillion-Token Pre-Training Data Pipeline

```text
Raw Web Ingestion (Common Crawl, ArXiv, GitHub, Wikipedia: 50+ TB)
   │
   ▼
[1. Text Extraction & Normalization]
   ├── Trafilatura / Resiliparse HTML stripping
   ├── NFC Unicode normalization & whitespace collapse
   │
   ▼
[2. Heuristic Quality Filtering]
   ├── Word count filters (50 < words < 100,000)
   ├── Punctuation & symbol ratio (< 0.15 non-alphanumeric)
   ├── Language ID (FastText classifier: drop if confidence < 0.65)
   │
   ▼
[3. Statistical & Model-Based Quality Filtering]
   ├── Perplexity Filter: Score text using a fast 1B reference LM (Drop high-loss garbage)
   ├── Document Quality Classifier: Logistic/DeBERTa trained on Wikipedia vs CommonCrawl
   │
   ▼
[4. Deduplication Engine]
   ├── Exact Substring Deduplication (Suffix arrays)
   └── Fuzzy Near-Duplicate Removal (MinHash LSH at Jaccard threshold >= 0.80)
   │
   ▼
[5. Decontamination & PII Scrubbing]
   ├── Presidio PII replacement (Emails, SSNs, IP addresses)
   └── 13-gram overlap removal against eval sets (MMLU, HumanEval, GSM8K)
   │
   ▼
Tokenization & Packing into Binary Bin/Idx Files (BPE 128K Vocabulary)
```

---

## 2️⃣ Fuzzy Deduplication: MinHash & Locality-Sensitive Hashing (LSH) Math

### Why Deduplication Is Mandatory:
If duplicate or near-duplicate web documents enter the training set:
1. The model memorizes exact phrases, degrading generalization.
2. The model enters **degenerate repetition loops** at inference time.
3. Up to 40% of compute budget is wasted learning the exact same sentences.

### How MinHash + LSH Works Mathematically:

```text
Document A: "The quick brown fox jumps" ──► Shingles (k=5): {"The q", "he qu", "e qui", ...}
                                                                  │
                                            Compute h(s) with M hash functions
                                                                  │
MinHash Signature Vector: [ h_1(min), h_2(min), ..., h_M(min) ] (e.g., M = 128)
                                │
Partition into b Bands of r Rows (M = b × r)
                                │
Hash each Band into a Bucket ──► Documents colliding in ANY bucket are Candidate Pairs!
```

1. **$k$-Shingling**: Split documents into overlapping character sequences of length $k$ (e.g., $k=5$).
2. **Jaccard Similarity**: 
   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
3. **MinHash**: Applying $M$ independent universal hash functions $h_1, \dots, h_M$ to all shingles. The probability that the minimum hash value matches equals the Jaccard similarity:
   $$P(\min(h(A)) = \min(h(B))) = J(A, B)$$
4. **LSH Banding**: Split the $M$-length signature into $b$ bands of $r$ rows ($M = b \cdot r$).
   - Probability that two documents with similarity $s$ collide in at least one band:
     $$P(\text{Collision}) = 1 - (1 - s^r)^b$$
   - Tuning $b$ and $r$ creates an S-curve threshold at $s \approx (1/b)^{1/r}$ (e.g., $s \ge 0.80$). Candidate pairs are checked in $O(1)$ hash table lookups rather than $O(N^2)$ brute-force comparisons!

---

## 3️⃣ Synthetic Data Generation: Evol-Instruct, Self-Instruct & UltraFeedback

When human domain experts are too expensive (\$100/hr for doctors or senior software engineers), top labs use **Frontier LLMs to Synthesize and Mutate Instruction Data**.

```text
Seed Instructions (1,000 hand-crafted simple prompts)
   │
   ▼
[Evol-Instruct Mutation Engine]
   ├── Mutation 1: In-Depth Evolution (Add complexity, multi-step constraints)
   ├── Mutation 2: In-Breadth Evolution (Create adjacent domain topics)
   ├── Mutation 3: Concretizing (Turn abstract request into specific scenario)
   └── Mutation 4: Elimination (Inject deliberate bugs/edge cases)
   │
   ▼
[Automated Elimination Filter]
   ├── Did the prompt become incomprehensible?
   └── Does the response fail to execute or contain hallucinations?
   │
   ▼
High-Complexity Synthetic Dataset (100,000+ Verified Instruction-Response Pairs)
```

### The 4 Evol-Instruct Mutation Directives:
1. **Add Constraints**: *"Rewrite this prompt to require that the solution uses no external libraries and executes in $O(N)$ time."*
2. **Deepening**: *"Deepen this prompt by asking for the underlying memory-allocation and cache-invalidation mechanics."*
3. **Concretizing**: *"Replace this abstract request with a real-world enterprise scenario involving HIPAA compliance."*
4. **Multi-Step Reasoning**: *"Expand this question to require 3 sequential dependent mathematical calculations."*

---

## 4️⃣ Post-Training Rejection Sampling & Process Reward Models (PRMs)

```text
Prompt (e.g., Complex LeetCode Problem)
   │
   ├──► Generate K = 16 Candidate Responses via High-Temperature Sampling
   │
   ▼
[Execution & Verification Gate]
   ├── Deterministic Test Harness (Python `pytest` execution in Docker Sandbox)
   │     • Passed all unit tests? ──► Candidate Accepted
   │     • Threw SyntaxError / Timeout? ──► Discarded
   │
   └── Process Reward Model (PRM)
         • Evaluates reasoning step-by-step: [Step 1: OK, Step 2: OK, Step 3: FAIL]
   │
   ▼
Selected "Golden" Response ──► Appended to Supervised Fine-Tuning (SFT) / DPO Buffer
```

### PRM (Process Reward Model) vs. ORM (Outcome Reward Model):
- **Outcome Reward Model (ORM)**: Evaluates only the final answer at the very end. 
  - *Failure Mode*: "Right answer for the wrong reason" (false positives in intermediate reasoning).
- **Process Reward Model (PRM)**: Scores every individual reasoning step $\text{Step}_1, \dots, \text{Step}_t$ with a reward $r_t \in [0, 1]$.
  - Powers reasoning models (like OpenAI o1/o3, DeepSeek-R1).
  - Enables **Monte Carlo Tree Search (MCTS)** at inference time to backtrack from flawed reasoning branches.

---

## 5️⃣ Benchmark Decontamination & Test-Set Leakage Prevention

> [!CAUTION]
> **The Data Contamination Scandal**: If a model's pre-training or fine-tuning set accidentally includes questions from **GSM8K, MMLU, HumanEval, or ARC**, its benchmark scores will be artificially inflated, leading to immediate public embarrassment and paper retractions.

### The Standard Decontamination Protocol:
1. **$N$-Gram Exact Overlap Filtering**:
   - For all benchmark questions, compute sliding 8-grams, 10-grams, and 13-grams.
   - Scan the candidate pre-training dataset. If any document shares a 13-gram overlap with an evaluation test prompt, the entire document is flagged and purged.
2. **Normalized Semantic Overlap**:
   - Strip whitespace, punctuation, convert to lowercase, and compute token-level Jaccard similarity.
3. **Synthetic Paraphrase Screening**:
   - Check whether seed prompts used in Evol-Instruct synthetic data generation share cosine similarity $\ge 0.85$ with any public benchmark questions.

---

## 6️⃣ Top 3 Data Engineering Interview Scenarios & Model Answers

### Scenario 1: *"How would you curate a dataset to train a 7B code model that beats models 2x its size?"*
**Verbal Answer Strategy:**
> "I would focus on **execution-verified synthetic data rather than raw GitHub scraping**:
> 1. Raw GitHub code is 70% boilerplate, dead code, and unrunnable snippets.
> 2. I would scrape GitHub exclusively for repositories with active CI/CD and passing unit tests.
> 3. I would build a synthetic generation loop: Take docstrings, prompt a teacher model to generate unit tests, and generate 5 candidate implementations.
> 4. Run the generated code through an air-gapped Docker sandbox. **Keep only the code that passes all unit tests, linters, and type checkers**.
> 5. A dataset of 200,000 100% verified, executed code pairs outperforms 10 million noisy scraped lines every time."

---

### Scenario 2: *"Your model passes your evaluation set with 92% accuracy, but beta users say it is terrible at reasoning. What happened?"*
**Verbal Answer Strategy:**
> "This points directly to **Eval Set Leakage or Narrow Distribution Overfitting**:
> 1. Our evaluation dataset likely contaminated the training pipeline, causing the model to memorize the test questions rather than learn generalized deduction.
> 2. To diagnose this, I would run an $n$-gram decontamination audit between the eval set and the training corpus.
> 3. Second, I would immediately test the model on **perturbed variations** of the eval set (e.g., changing numerical constants or variable names in math problems). If accuracy drops from 92% to 30% when numbers change, the model is memorizing, not reasoning."

---

### Scenario 3: *"How do you handle multilingual data balance when English accounts for 85% of your web scrape?"*
**Verbal Answer Strategy:**
> "Simply training on raw proportions will starve low-resource languages, leading to poor tokenization compression and token degradation.
> 1. **Upsampling via Temperature Scaling**: We smooth language distribution by sampling with a temperature $\alpha \in (0, 1)$:
>    $$P(L_i) \propto (N_i)^\alpha$$
>    where $\alpha = 0.3$ downsamples English and boosts low-resource languages.
> 2. **Targeted Translation & Synthetic Expansion**: Translate high-quality English educational and STEM data into target languages using frontier models, ensuring terminology consistency.
> 3. **BPE Vocabulary Allocation**: Ensure the tokenizer allocates dedicated token slots to non-Latin scripts (Devanagari, Cyrillic, CJK) to avoid token explosion where a single word takes 8 tokens."
