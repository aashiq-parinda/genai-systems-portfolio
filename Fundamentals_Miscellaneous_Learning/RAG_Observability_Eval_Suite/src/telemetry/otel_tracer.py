import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

# OpenTelemetry GenAI Semantic Conventions Constants
# Based on OpenTelemetry Semantic Conventions for Generative AI systems
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# RAG Specific Semantic Conventions
GEN_AI_RETRIEVAL_QUERY = "gen_ai.retrieval.query"
GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
GEN_AI_RETRIEVAL_CHUNKS_RETURNED = "gen_ai.retrieval.chunks_returned"
GEN_AI_RETRIEVAL_TOP_SCORE = "gen_ai.retrieval.top_score"
GEN_AI_RETRIEVAL_SOURCES = "gen_ai.retrieval.sources"

# Tool Execution Semantic Conventions
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_ARGS = "gen_ai.tool.arguments"
GEN_AI_TOOL_STATUS = "gen_ai.tool.status"
GEN_AI_TOOL_ERROR = "gen_ai.tool.error"

@dataclass
class OTelSpan:
    """Represents an OpenTelemetry Span with GenAI semantic convention attributes."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "UNSET" # OK, ERROR, UNSET
    status_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def finish(self, status: str = "OK", status_message: Optional[str] = None):
        self.end_time = time.time()
        self.status = status
        self.status_message = status_message

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return round((time.time() - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events
        }

class OTelTracer:
    """
    OpenTelemetry Tracer for Generative AI pipelines.
    Maintains active traces, spans hierarchy, and exports to OTel buffers.
    """
    def __init__(self, service_name: str = "cloudflow-rag-service"):
        self.service_name = service_name
        self.spans_buffer: List[OTelSpan] = []
        self._active_trace_id: Optional[str] = None
        self._span_stack: List[OTelSpan] = []

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        self._active_trace_id = trace_id or uuid.uuid4().hex
        self._span_stack = []
        return self._active_trace_id

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self._active_trace_id:
            self.start_trace()

        parent_span_id = self._span_stack[-1].span_id if self._span_stack else None
        span = OTelSpan(
            name=name,
            trace_id=self._active_trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        self._span_stack.append(span)
        try:
            yield span
            if span.status == "UNSET":
                span.finish("OK")
        except Exception as exc:
            span.finish("ERROR", str(exc))
            span.add_event("exception", {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc)
            })
            raise
        finally:
            self._span_stack.pop()
            self.spans_buffer.append(span)

    def get_spans_for_trace(self, trace_id: str) -> List[OTelSpan]:
        return [s for s in self.spans_buffer if s.trace_id == trace_id]

    def clear(self):
        self.spans_buffer.clear()
        self._span_stack.clear()
        self._active_trace_id = None

# Global default tracer
global_tracer = OTelTracer()
