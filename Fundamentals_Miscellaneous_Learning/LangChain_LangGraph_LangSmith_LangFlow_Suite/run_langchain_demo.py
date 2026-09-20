#!/usr/bin/env python3
"""
LangChain Fundamentals Hands-On Demo.
Demonstrates:
1. LCEL Piping: PromptTemplate | ChatModel | JsonOutputParser
2. RunnableParallel: Multi-dimensional extraction in parallel
3. Resilient Model Failover: with_fallbacks()
4. Memory Management: Message trimming & Conversation History Buffer
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import Config
from src.langchain_core.lcel_pipeline import (
    build_lcel_intent_classifier,
    build_lcel_parallel_analyzer,
    build_lcel_resilient_fallback_chain,
    IntentClassification
)
from src.langchain_core.memory_and_history import (
    ConversationBuffer,
    trim_messages,
    SummaryMemoryManager
)

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

def demo_lcel_pipe():
    print_header("1. Modern LCEL Pipe Composition (prompt | model | parser)")
    chain = build_lcel_intent_classifier()
    
    test_queries = [
        "How much does a monthly dedicated Qdrant cluster cost with 10M vectors?",
        "Our ingestion worker crashed with SIGSEGV out of memory during index build."
    ]
    
    for q in test_queries:
        print(f"\n[Input Query]: \"{q}\"")
        output: IntentClassification = chain.invoke({"text": q})
        print(f" -> Output Type: {type(output).__name__}")
        print(f" -> Intent:      {output.intent}")
        print(f" -> Confidence:  {output.confidence:.2f}")
        print(f" -> Routing To:  {output.target_handler}")

def demo_runnable_parallel():
    print_header("2. RunnableParallel (Multi-Aspect Analysis)")
    parallel_chain = build_lcel_parallel_analyzer()
    query = {"text": "URGENT: Pinecone latency spiked to 450ms after deploying new embedding model."}
    
    print(f"[Input Query]: \"{query['text']}\"\n")
    results = parallel_chain.invoke(query)
    
    print(json.dumps(results, indent=2))

def demo_fallbacks():
    print_header("3. Resilient Failover with .with_fallbacks()")
    resilient_chain = build_lcel_resilient_fallback_chain()
    query = {"text": "Perform high-throughput batch vector indexing."}
    
    print(f"[Input Query]: \"{query['text']}\"")
    print(" -> Primary model: gpt-4o (configured to encounter a simulated 429 RateLimit error)")
    print(" -> Fallback model: gpt-4o-mini (automatic failover route)")
    
    response = resilient_chain.invoke(query)
    print(f"\n[Final Response]: {response}")

def demo_memory_trimming():
    print_header("4. Memory Buffer & Sliding Window Trimming")
    buffer = ConversationBuffer()
    buffer.add_system_message("You are an expert enterprise LLM architect.")
    buffer.add_user_message("What is the difference between dense and sparse vectors?")
    buffer.add_ai_message("Dense vectors capture semantic meaning; sparse vectors capture exact lexical terms like BM25.")
    buffer.add_user_message("Which one is better for hybrid search?")
    buffer.add_ai_message("Combining both via reciprocal rank fusion (RRF) delivers the best recall.")
    buffer.add_user_message("Can Qdrant handle both?")
    buffer.add_ai_message("Yes, Qdrant natively supports dense, sparse, and multi-vector search.")
    
    raw_msgs = buffer.get_messages()
    print(f"Total raw conversation turns: {len(raw_msgs)} messages")
    
    trimmed = trim_messages(raw_msgs, max_messages=3, keep_system=True)
    print(f"Trimmed to window (System + last 3 turns): {len(trimmed)} messages")
    for m in trimmed:
        print(f" - [{m.role.upper()}]: {m.content[:55]}...")

if __name__ == "__main__":
    print("\n🚀 Starting LangChain LCEL & Memory Demonstrations...")
    print(f"Configuration: {Config.summary()}")
    demo_lcel_pipe()
    demo_runnable_parallel()
    demo_fallbacks()
    demo_memory_trimming()
    print("\n✅ LangChain Demonstrations Completed Successfully!\n")
