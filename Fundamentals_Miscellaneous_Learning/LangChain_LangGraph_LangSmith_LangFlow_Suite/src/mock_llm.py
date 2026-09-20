"""
High-Fidelity Mock LLM Engine.
Provides deterministic, structured, and realistic responses for LangChain and LangGraph
experiments without requiring paid external API credentials.
"""

import time
import json
import random
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class MockUsage(BaseModel):
    prompt_tokens: int = 42
    completion_tokens: int = 128
    total_tokens: int = 170

class MockAIMessage(BaseModel):
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "metadata": self.response_metadata
        }

class MockLLM:
    """
    Simulated Chat Model with configurable latency, token tracking, and structured synthesis.
    """
    def __init__(self, model_name: str = "mock-gpt-4o", latency_ms: int = 120):
        self.model_name = model_name
        self.latency_ms = latency_ms

    def invoke(self, prompt_or_messages: Any, **kwargs) -> MockAIMessage:
        time.sleep(self.latency_ms / 1000.0)

        prompt_str = str(prompt_or_messages)

        # 1. Routing / Intent Classification Prompt
        if "classify" in prompt_str.lower() or "intent" in prompt_str.lower():
            if "pricing" in prompt_str.lower() or "cost" in prompt_str.lower():
                content = json.dumps({
                    "intent": "billing_inquiry",
                    "confidence": 0.96,
                    "target_handler": "billing_specialist"
                })
            elif "error" in prompt_str.lower() or "bug" in prompt_str.lower() or "fail" in prompt_str.lower():
                content = json.dumps({
                    "intent": "technical_support",
                    "confidence": 0.98,
                    "target_handler": "devops_specialist"
                })
            else:
                content = json.dumps({
                    "intent": "general_knowledge",
                    "confidence": 0.91,
                    "target_handler": "general_rag_handler"
                })

        # 2. Evaluation / Judge Prompt
        elif "evaluate" in prompt_str.lower() or "groundedness" in prompt_str.lower() or "judge" in prompt_str.lower():
            content = json.dumps({
                "faithfulness_score": 0.94,
                "relevance_score": 0.92,
                "hallucination_detected": False,
                "reasoning": "The synthesis directly matches facts from retrieved context chunks without fabrication."
            })

        # 3. LangGraph Agent Planning Prompt
        elif "plan" in prompt_str.lower() and "subtasks" in prompt_str.lower():
            content = json.dumps({
                "plan": [
                    "1. Fetch vector database comparison benchmarks (Pinecone vs Milvus vs Qdrant)",
                    "2. Extract ingestion throughput, hybrid search, and filtering metrics",
                    "3. Synthesize structured report with technical trade-offs",
                    "4. Submit draft for human review and validation"
                ],
                "confidence": 0.95
            })

        # 4. LangGraph Research / Tool Execution Synthesis
        elif "research" in prompt_str.lower() or "findings" in prompt_str.lower():
            content = (
                "Verified Research Findings:\n"
                "- Qdrant: Written in Rust, exceptional payload filtering performance, native HNSW + quantization.\n"
                "- Pinecone: Managed serverless vector index with instant scaling and metadata-based namespace partitioning.\n"
                "- Milvus: Distributed architecture suited for multi-billion vector clusters with Kubernetes operator."
            )

        # 5. LangGraph Critique / Revision Check
        elif "critique" in prompt_str.lower() or "review draft" in prompt_str.lower():
            content = json.dumps({
                "quality_score": 0.88,
                "critique": "Comprehensive architectural points. Ready for human verification.",
                "requires_revision": False
            })

        # Default synthesis
        else:
            content = (
                f"[{self.model_name}] Synthesized response: Based on the provided context, "
                f"the operational requirements have been validated. Total latency: {self.latency_ms}ms."
            )

        usage = MockUsage(
            prompt_tokens=max(20, len(prompt_str.split())),
            completion_tokens=len(content.split()),
            total_tokens=max(20, len(prompt_str.split())) + len(content.split())
        )

        return MockAIMessage(
            content=content,
            response_metadata={
                "model_name": self.model_name,
                "usage": usage.model_dump(),
                "finish_reason": "stop"
            }
        )

    def stream(self, prompt_or_messages: Any, **kwargs):
        """Simulate token-by-token streaming."""
        full_msg = self.invoke(prompt_or_messages, **kwargs)
        tokens = full_msg.content.split(" ")
        for token in tokens:
            time.sleep(0.015)
            yield f"{token} "
