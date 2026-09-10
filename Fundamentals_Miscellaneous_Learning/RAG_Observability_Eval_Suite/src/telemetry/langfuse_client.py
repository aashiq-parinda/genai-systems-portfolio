import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ..config import settings
from .otel_tracer import OTelSpan

@dataclass
class LangfuseScore:
    name: str
    value: float
    comment: Optional[str] = None
    trace_id: Optional[str] = None
    observation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class LangfuseTelemetryClient:
    """
    Client for Langfuse observability: handles prompt version management,
    trace tree ingestion, span generation, and custom evaluation metric scores.
    """
    def __init__(self):
        self.host = settings.langfuse_host
        self.public_key = settings.langfuse_public_key
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.scores: List[LangfuseScore] = []
        self.prompts: Dict[str, PromptVersion] = {}
        self._init_prompts()

    def _init_prompts(self):
        """Initialize version-controlled prompts in Langfuse registry."""
        self.prompts["support_rag_system"] = PromptVersion(
            name="support_rag_system",
            version="v1.2.0",
            template=(
                "You are an expert customer support agent for CloudFlow SaaS.\n"
                "Answer the user's question using ONLY the retrieved context below.\n"
                "If the context does not contain the answer, politely state that you cannot answer from documentation.\n"
                "If the user asks about an order or account status, specify if a tool call is needed.\n\n"
                "Retrieved Context:\n{retrieved_context}\n\n"
                "User Inquiry: {user_query}"
            ),
            variables=["retrieved_context", "user_query"],
            metadata={"domain": "customer_support", "strict_grounding": True}
        )

    def get_prompt(self, name: str) -> PromptVersion:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found in Langfuse prompt registry.")
        return self.prompts[name]

    def create_trace(
        self,
        trace_id: str,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "name": name,
            "user_id": user_id or "anonymous_user",
            "session_id": session_id or "session_default",
            "tags": tags or ["rag", "production"],
            "metadata": metadata or {},
            "spans": [],
            "scores": [],
            "start_time": time.time(),
            "end_time": None
        }

    def sync_otel_spans(self, trace_id: str, spans: List[OTelSpan]):
        """Convert OpenTelemetry GenAI spans into Langfuse observation tree."""
        if trace_id not in self.traces:
            self.create_trace(trace_id=trace_id, name="rag_query_execution")

        trace_data = self.traces[trace_id]
        for span in spans:
            observation_type = "span"
            if "gen_ai.usage.completion_tokens" in span.attributes:
                observation_type = "generation"

            lf_span = {
                "id": span.span_id,
                "parent_id": span.parent_span_id,
                "name": span.name,
                "type": observation_type,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "status_message": span.status_message,
                "attributes": span.attributes,
                "events": span.events
            }
            trace_data["spans"].append(lf_span)
        trace_data["end_time"] = time.time()

    def record_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        observation_id: Optional[str] = None
    ):
        """Log custom evaluation score (groundedness, context relevance) back to Langfuse."""
        score = LangfuseScore(
            trace_id=trace_id,
            observation_id=observation_id,
            name=name,
            value=round(value, 3),
            comment=comment
        )
        self.scores.append(score)
        if trace_id in self.traces:
            self.traces[trace_id]["scores"].append({
                "name": score.name,
                "value": score.value,
                "comment": score.comment,
                "timestamp": score.timestamp
            })

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        return self.traces.get(trace_id, {})

langfuse_client = LangfuseTelemetryClient()
