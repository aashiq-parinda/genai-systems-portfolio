import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ..config import settings
from .otel_tracer import OTelSpan

# OpenInference Semantic Conventions for Arize Phoenix
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
SPAN_KIND_RETRIEVER = "RETRIEVER"
SPAN_KIND_LLM = "LLM"
SPAN_KIND_TOOL = "TOOL"
SPAN_KIND_CHAIN = "CHAIN"

RETRIEVAL_DOCUMENTS = "retrieval.documents"
LLM_INPUT_MESSAGES = "llm.input_messages"
LLM_OUTPUT_MESSAGES = "llm.output_messages"

@dataclass
class PhoenixRAGTrace:
    trace_id: str
    project_name: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    evaluations: Dict[str, Any] = field(default_factory=dict)

class PhoenixTelemetryClient:
    """
    Client for Arize Phoenix: tracks RAG retrieval relevance, context drift,
    and formats spans using OpenInference semantic conventions.
    """
    def __init__(self):
        self.endpoint = settings.phoenix_collector_endpoint
        self.project_name = settings.phoenix_project_name
        self.traces: Dict[str, PhoenixRAGTrace] = {}

    def log_trace(self, trace_id: str, spans: List[OTelSpan], query: str, retrieved_chunks: List[Any]):
        """Convert spans to Phoenix OpenInference format and record RAG retrieval metrics."""
        phoenix_spans = []
        retrieval_scores = []
        
        for span in spans:
            # Map OTel spans to Phoenix OpenInference kinds
            kind = SPAN_KIND_CHAIN
            if "retrieve" in span.name.lower():
                kind = SPAN_KIND_RETRIEVER
            elif "chat" in span.name.lower() or "llm" in span.name.lower():
                kind = SPAN_KIND_LLM
            elif "tool" in span.name.lower():
                kind = SPAN_KIND_TOOL

            p_span = {
                "name": span.name,
                "span_id": span.span_id,
                "parent_id": span.parent_span_id,
                "trace_id": span.trace_id,
                "kind": kind,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "attributes": {
                    OPENINFERENCE_SPAN_KIND: kind,
                    **span.attributes
                }
            }
            
            # Enrich retriever span with Phoenix document payloads
            if kind == SPAN_KIND_RETRIEVER and retrieved_chunks:
                p_span["attributes"][RETRIEVAL_DOCUMENTS] = [
                    {
                        "document.id": getattr(chunk, "chunk_id", f"chunk_{i}"),
                        "document.content": getattr(chunk, "content", str(chunk))[:200] + "...",
                        "document.score": getattr(chunk, "score", 0.0),
                        "document.metadata": getattr(chunk, "metadata", {})
                    }
                    for i, chunk in enumerate(retrieved_chunks)
                ]
                retrieval_scores = [getattr(c, "score", 0.0) for c in retrieved_chunks]

            phoenix_spans.append(p_span)

        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        
        self.traces[trace_id] = PhoenixRAGTrace(
            trace_id=trace_id,
            project_name=self.project_name,
            spans=phoenix_spans,
            retrieval_metrics={
                "query": query,
                "chunks_retrieved_count": len(retrieved_chunks),
                "avg_retrieval_similarity": round(avg_retrieval_score, 4),
                "top_score": max(retrieval_scores) if retrieval_scores else 0.0
            }
        )

    def log_evaluation(self, trace_id: str, evaluation_name: str, score: float, explanation: str):
        if trace_id in self.traces:
            self.traces[trace_id].evaluations[evaluation_name] = {
                "score": score,
                "explanation": explanation,
                "timestamp": time.time()
            }

    def get_trace_summary(self, trace_id: str) -> Optional[PhoenixRAGTrace]:
        return self.traces.get(trace_id)

phoenix_client = PhoenixTelemetryClient()
