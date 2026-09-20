"""
LangSmith Observability & Tracing Engine.
Demonstrates:
1. Tracing Configuration & Environment Setup (LANGCHAIN_TRACING_V2)
2. `@traceable` decorator capturing nested parent-child run trees
3. Latency, Token Counts, Input/Output Payloads, and Error Boundaries
4. Formatting Traces into a LangSmith-Style Span Waterfall
"""

import time
import uuid
import functools
from typing import Dict, Any, List, Optional, Callable
from ..config import Config

class RunSpan:
    """Represents a single span/run in the LangSmith run tree."""
    def __init__(self, name: str, run_type: str = "chain", parent_run_id: Optional[str] = None, tags: Optional[List[str]] = None):
        self.run_id = str(uuid.uuid4())[:8]
        self.name = name
        self.run_type = run_type  # 'llm', 'chain', 'tool', 'retriever', 'agent'
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.latency_ms: float = 0.0
        self.tokens: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self.error: Optional[str] = None
        self.children: List["RunSpan"] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "run_type": self.run_type,
            "parent_run_id": self.parent_run_id,
            "latency_ms": round(self.latency_ms, 2),
            "tokens": self.tokens,
            "tags": self.tags,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "children": [c.to_dict() for c in self.children]
        }

class TraceContext:
    """Manages active span nesting during execution."""
    _current_span: Optional[RunSpan] = None
    _completed_traces: List[RunSpan] = []

    @classmethod
    def start_span(cls, name: str, run_type: str = "chain", tags: Optional[List[str]] = None) -> RunSpan:
        parent_id = cls._current_span.run_id if cls._current_span else None
        span = RunSpan(name=name, run_type=run_type, parent_run_id=parent_id, tags=tags)
        span.start_time = time.time()

        if cls._current_span:
            cls._current_span.children.append(span)
        else:
            cls._completed_traces.append(span)

        span._previous_active = cls._current_span
        cls._current_span = span
        return span

    @classmethod
    def end_span(cls, span: RunSpan, outputs: Any = None, error: Optional[Exception] = None):
        span.end_time = time.time()
        span.latency_ms = (span.end_time - span.start_time) * 1000.0
        if error:
            span.error = str(error)
        else:
            span.outputs = {"result": outputs} if not isinstance(outputs, dict) else outputs

        # Calculate simulated or captured tokens
        if span.run_type == "llm":
            span.tokens = {"prompt": 45, "completion": 85, "total": 130}

        cls._current_span = getattr(span, "_previous_active", None)

    @classmethod
    def get_latest_trace(cls) -> Optional[RunSpan]:
        return cls._completed_traces[-1] if cls._completed_traces else None

def traceable(name: Optional[str] = None, run_type: str = "chain", tags: Optional[List[str]] = None):
    """
    Decorator mirroring langsmith.run_helpers.traceable.
    Automatically wraps function execution in a parent or child trace span.
    """
    def decorator(func: Callable):
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span = TraceContext.start_span(span_name, run_type=run_type, tags=tags)
            span.inputs = {"args": [str(a) for a in args], "kwargs": kwargs}
            try:
                result = func(*args, **kwargs)
                TraceContext.end_span(span, outputs=result)
                return result
            except Exception as e:
                TraceContext.end_span(span, error=e)
                raise e
        return wrapper
    return decorator

def render_trace_waterfall(root_span: RunSpan, indent: int = 0) -> str:
    """Renders an ASCII trace waterfall view matching LangSmith's trace inspector."""
    prefix = "  " * indent
    connector = "└─ " if indent > 0 else "• "
    type_badge = f"[{root_span.run_type.upper():^9}]"
    latency_str = f"{root_span.latency_ms:>6.1f}ms"
    token_str = f"{root_span.tokens['total']} tok" if root_span.tokens['total'] else "      -"
    
    line = f"{prefix}{connector}{type_badge} {root_span.name:<32} {latency_str} | {token_str} | (id: {root_span.run_id})"
    lines = [line]
    for child in root_span.children:
        lines.append(render_trace_waterfall(child, indent + 1))
    return "\n".join(lines)
