# Contract Risk Review — LLM Fine-Tuning + Agentic Reasoning

An end-to-end AI system that automates the first pass of vendor contract risk
review: fine-tunes a small language model to classify contract clauses,
retrieves similar precedent clauses via RAG, generates plain-English risk
explanations through an agentic pipeline, and routes low-confidence
predictions to a human reviewer.

## 📺 Video Walkthrough & Live Demo

[![Watch the Video Walkthrough](https://img.youtube.com/vi/-iob4zi4bfY/maxresdefault.jpg)](https://www.youtube.com/watch?v=-iob4zi4bfY)

> 💡 *Click the thumbnail above to watch the contract risk classifier walkthrough, LoRA fine-tuning benchmarks, and agentic review pipeline on YouTube.*

---

## Problem

Legal and compliance teams manually read every clause of every vendor
contract to catch risky terms — unlimited liability, weak termination
rights, auto-renewal traps, one-sided indemnification. It's slow,
inconsistent across reviewers, and doesn't scale with deal volume.

## What this does

Given a contract's text, the pipeline:

1. Splits it into individual clauses
2. Classifies each clause into a legal category (Indemnification, Governing
   Law, Termination, etc.) using a fine-tuned classifier
3. Maps that category to a risk tier (High / Medium / Low)
4. Retrieves the most similar known precedent clause for context
5. Generates a 2-sentence plain-English explanation of the risk
6. Flags any low-confidence prediction for human review instead of
   auto-finalizing it

Output is a clause-by-clause report plus a review queue — not a black-box
"risky/not risky" label.

## Architecture

```
Contract text
     │
     ▼
[ split into clauses ]
     │
     ▼
[ LoRA-fine-tuned DeBERTa-v3 classifier ] ──► category + confidence
     │
     ▼
[ Chroma vector DB ] ──► retrieve nearest precedent clause (RAG)
     │
     ▼
[ Qwen2.5-1.5B-Instruct ] ──► plain-English risk explanation
     │
     ▼
[ Guardrail: confidence threshold ] ──► auto-finalize  OR  human review queue
     │
     ▼
Clause-by-clause report
```

Orchestration is a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine, not a single model call — each clause loops through
classify → retrieve → explain → guardrail-check before moving to the next.

## Topics covered

| Area | What was actually built |
|---|---|
| **Parameter-efficient fine-tuning** | LoRA adapters on `microsoft/deberta-v3-small`, trained on [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (~100 real contract clause categories from SEC filings) |
| **Retrieval-Augmented Generation** | Chroma vector DB of precedent clauses, embedded with `sentence-transformers`, queried at inference time to ground each explanation in a real example |
| **Agentic orchestration** | LangGraph state machine: split → classify (loop) → retrieve → summarize → guardrail-check |
| **Guardrails** | (1) Confidence-threshold human-in-the-loop routing, (2) per-category fairness/bias report (precision/recall/F1 by clause type), (3) explainability via top-3 alternative predictions + retrieved precedent |
| **Evaluation** | Accuracy, macro-F1, per-category breakdown, inference latency (mean/p50/p95), and a time-savings estimate — explicitly labeled as resting on a stated assumption, not measured fact |
| **Deployment path** | LoRA adapter pushed to the Hugging Face Hub, structured for serving from a Hugging Face Space |

## Why these design choices

- **Small encoder model for classification, not the LLM itself.** Clause-type
  classification doesn't need generation. A small encoder is faster, cheaper
  to run, and easier to evaluate rigorously. The LLM is reserved for the one
  step that actually needs generation — the plain-English summary.
- **LEDGAR over ContractNLI.** The original scope called for classifying
  *risky* clauses. ContractNLI classifies entailment about confidentiality
  hypotheses — unrelated to the task. LEDGAR is real clause-category
  classification, which a risk-tier mapping can sit on top of.
- **RAG grounds the explanation, not just the label.** A bare "this is
  Indemnification" is less useful than showing the closest real precedent it
  resembles — it gives a reviewer something concrete to check the model's
  reasoning against.
- **Guardrails are functional, not decorative.** Low-confidence predictions
  are routed to a queue rather than silently trusted. Fairness is checked as
  actual per-category F1 numbers, not asserted in prose.

## Honest limitations

- LEDGAR's labels come from SEC filings, a reasonable but imperfect proxy for
  what a legal team would consider "risky" — a real deployment would want
  legal-expert-labeled data.
- Clause splitting is a simple paragraph split; real contracts vary enough in
  formatting that a proper legal-clause segmenter would do better.
- The latency benchmark measures the classifier only, not the full pipeline
  including the LLM summary step, which is meaningfully slower. Any
  end-to-end latency claim should be benchmarked separately.
- The fairness check is a proxy (per-category model performance), not an
  audit for bias against protected classes.
- The time-savings estimate rests on an assumed manual-review time per
  clause — it's a placeholder to replace with a real team's numbers, not a
  measured result.

## Repo contents

- `Contract_Risk_Review_Fixed.ipynb` — the full notebook: data loading,
  fine-tuning, RAG setup, agentic pipeline, guardrails, evaluation
- `README.md` — this file

## Running it

1. Open the notebook in Google Colab
2. `Runtime > Change runtime type` → **T4 GPU**
3. Add a Hugging Face token as a Colab Secret named `HF_TOKEN` (key icon,
   left sidebar) with write access, and enable notebook access
4. Run cells top to bottom

Training LoRA on `deberta-v3-small` over LEDGAR takes roughly 10–20 minutes
on a T4. The agentic pipeline demo runs on a small hardcoded sample contract
at the end of the notebook.

## Talking about this in an interview

**One-line pitch:** "I built an end-to-end AI system that automates
contract risk review — it fine-tunes a small language model to classify
contract clauses, then wraps that in an agentic pipeline that retrieves
similar precedent clauses, generates plain-English risk explanations, and
flags low-confidence predictions for human review, all evaluated on real
accuracy, latency, and fairness metrics."

**If asked "why not fine-tune a bigger model / the LLM itself?"** — the task
is clause-type classification, not generation; a small encoder is faster,
cheaper, and easier to evaluate than an LLM, which is reserved for the part
that actually needs generation.

**If asked "how do you know it's not biased?"** — point to the per-category
fairness report; be upfront that it's a proxy check on classifier
performance by category, not an audit for bias against protected classes.

**If asked "what would you do with more time?"** — real legal-expert-labeled
data instead of LEDGAR's SEC-filing labels, a better clause segmenter, and
end-to-end latency benchmarking including the LLM step.

**On the word "agentic"** — be precise if pushed: this is a defined,
multi-step, stateful graph with conditional routing (classify → guardrail →
loop), not a model autonomously choosing which tools to call. Don't oversell
it as more autonomous than it is.
